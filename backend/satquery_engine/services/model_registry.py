from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import httpx

from satquery_engine.config import settings
from satquery_engine.models.registry import LocalModelRegistry
from satquery_engine.services.vlm_provider import VLMConnector


@dataclass(frozen=True)
class ModelSpec:
    name: str
    purpose: str
    endpoint: str | None
    required_inputs: tuple[str, ...]
    version: str = "unverified"
    accepted_modalities: tuple[str, ...] = ()
    required_bands: tuple[str, ...] = ()
    input_type: str = "raster_paths"
    output_type: str = "provider_json"
    supported_tasks: tuple[str, ...] = ()
    expected_resolution: str = "not documented"
    hardware_requirements: str = "external inference service"
    checkpoint_path: str | None = None
    calibration_version: str | None = None
    training_data: str | None = None
    evaluation_metrics: dict | None = None
    license_provenance: str = "must be supplied by deployment"
    failure_modes: tuple[str, ...] = ("missing runtime", "out of distribution", "missing bands")

    @property
    def available(self) -> bool:
        return bool(self.endpoint)


class ModelRegistry:
    """Endpoint-backed lazy model registry with unified local path resolution.

    Heavy model weights are deliberately isolated from the API process. Each
    model server must expose POST /infer and return JSON with `answer`, optional
    `confidence`, and optional `evidence` fields.
    Local models are resolved authoritatively via ModelRegistry.get_path(model_id).
    """

    @classmethod
    def get_manifest_path(cls) -> Path:
        candidates = [
            settings.model_dir / "manifests/models.yaml",
            Path(__file__).resolve().parents[3] / "models/manifests/models.yaml",
            Path(__file__).resolve().parents[3] / "MODEL_MANIFEST.yaml",
        ]
        for c in candidates:
            if c.is_file():
                return c
        return candidates[0]

    @classmethod
    def load_models_yaml(cls) -> dict[str, Any]:
        manifest_file = cls.get_manifest_path()
        if not manifest_file.is_file():
            return {}
        try:
            import yaml
            with open(manifest_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("models", data)
        except Exception:
            return {}

    @classmethod
    def get_path(cls, model_id: str) -> Path | None:
        """Authoritative path resolver for any model checkpoint or directory."""
        clean_id = model_id.strip().lower()
        local = LocalModelRegistry(settings.model_dir, cls.get_manifest_path()).get(clean_id)
        if local is not None and local.local_path.exists():
            if local.local_path.is_file():
                return local.local_path
            verified = local.metadata.get("files_verified", [])
            if len(verified) == 1:
                return Path(verified[0])
            return local.local_path
        # 1. Check models.yaml entries
        models_data = cls.load_models_yaml()
        if isinstance(models_data, dict):
            for key, entry in models_data.items():
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get("model_id", "")).strip().lower()
                if clean_id in (key.lower(), entry_id, entry_id.split("/")[-1]):
                    local_path = entry.get("path") or entry.get("checkpoint_path")
                    if local_path:
                        p = Path(local_path)
                        if p.exists():
                            return p
                        # Try relative to settings.model_dir
                        rel_p = settings.model_dir / local_path
                        if rel_p.exists():
                            return rel_p
        elif isinstance(models_data, list):
            for entry in models_data:
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get("id", entry.get("model_id", ""))).strip().lower()
                if clean_id in (entry_id, entry_id.split("/")[-1]):
                    local_path = entry.get("checkpoint_path") or entry.get("path")
                    if local_path:
                        p = Path(local_path)
                        if p.exists():
                            return p
                        rel_p = settings.model_dir / local_path
                        if rel_p.exists():
                            return rel_p

        # 2. Known default path heuristics under settings.model_dir
        root = settings.model_dir
        path_map = {
            "dinov3s-buildings": root / "buildings/dinov3s-buildings/onnx/model.onnx",
            "building_primary": root / "buildings/dinov3s-buildings/onnx/model.onnx",
            "hotosm/dinov3s-buildings": root / "buildings/dinov3s-buildings/onnx/model.onnx",
            "geobase/dinov3s-buildings": root / "buildings/dinov3s-buildings/onnx/model.onnx",
            "geobase/building-detection": root / "buildings/geobase-building-detection/onnx/model_quantized.onnx",
            "building_secondary": root / "buildings/geobase-building-detection/onnx/model_quantized.onnx",
            "flair-rgb": root / "land_cover/flair-rgb-15cl-deeplabv3/FLAIR-INC_rgb_15cl_resnet34-deeplabv3_weights.pth",
            "flair": root / "land_cover/flair-rgb-15cl-deeplabv3/FLAIR-INC_rgb_15cl_resnet34-deeplabv3_weights.pth",
            "land_rgb": root / "land_cover/flair-rgb-15cl-deeplabv3/FLAIR-INC_rgb_15cl_resnet34-deeplabv3_weights.pth",
            "bigearthnet-s2": root / "land_cover/bigearthnet-s2-resnet50/model.safetensors",
            "land_s2": root / "land_cover/bigearthnet-s2-resnet50/model.safetensors",
            "bigearthnet-s1": root / "land_cover/bigearthnet-s1-resnet50/model.safetensors",
            "land_s1": root / "land_cover/bigearthnet-s1-resnet50/model.safetensors",
            "bigearthnet-s1s2": root / "land_cover/bigearthnet-s1s2-resnet101/model.safetensors",
            "land_s1s2": root / "land_cover/bigearthnet-s1s2-resnet101/model.safetensors",
            "prithvi": root / "water/prithvi-sen1floods11/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
            "water_s2": root / "water/prithvi-sen1floods11/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
            "earthdial": root / "vlm/earthdial-4b",
            "earthdial-4b-rgb": root / "vlm/earthdial-4b/EarthDial_4B_RGB",
            "earthdial-4b-ms": root / "vlm/earthdial-4b/EarthDial_4B_MS",
            "croma": root / "fusion/croma/CROMA_base.pt",
            "croma_base": root / "fusion/croma/CROMA_base.pt",
            "croma_large": root / "fusion/croma/CROMA_large.pt",
            "remoteclip": root / "remoteclip/rn50/RemoteCLIP-RN50.pt",
            "remoteclip_rn50": root / "remoteclip/rn50/RemoteCLIP-RN50.pt",
            "remoteclip_vit_b_32": root / "remoteclip/vit-b-32/RemoteCLIP-ViT-B-32.pt",
            "tinycd": root / "change_detection/tinycd/levir_best.pth",
            "bit": root / "change_detection/bit/bit_r18_256x256_40k_levircd.pth",
            "changerex": root / "change_detection/changerex/ChangerEx_r18-512x512_40k_levircd.pth",
            "ban": root / "change_detection/ban/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.pth",
        }
        for k, p in path_map.items():
            if clean_id in (k, k.replace("_", "-"), k.replace("-", "_")):
                if p.exists():
                    return p

        # 3. Fallback to existing legacy model files if present
        legacy_candidates = [
            root / "buildings/model.onnx",
            root / "croma/CROMA_base.pt",
            root / "remoteclip/RemoteCLIP-RN50.pt",
            root / "remoteclip/RemoteCLIP-ViT-B-32.pt",
            root / "earthdial/EarthDial_4B_RGB",
            root / "earthdial/EarthDial_4B_MS",
            root / "landcover/deeplabv3_landcover_4c.onnx",
        ]
        for c in legacy_candidates:
            if clean_id in str(c).lower() and c.exists():
                return c
        # 1. Check models.yaml entries
        models_data = cls.load_models_yaml()
        if isinstance(models_data, dict):
            for key, entry in models_data.items():
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get("model_id", "")).strip().lower()
                if clean_id in (key.lower(), entry_id, entry_id.split("/")[-1]):
                    local_path = entry.get("path") or entry.get("checkpoint_path")
                    if local_path:
                        p = Path(local_path)
                        if p.exists():
                            return p
                        # Try relative to settings.model_dir
                        rel_p = settings.model_dir / local_path
                        if rel_p.exists():
                            return rel_p
        elif isinstance(models_data, list):
            for entry in models_data:
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get("id", entry.get("model_id", ""))).strip().lower()
                if clean_id in (entry_id, entry_id.split("/")[-1]):
                    local_path = entry.get("checkpoint_path") or entry.get("path")
                    if local_path:
                        p = Path(local_path)
                        if p.exists():
                            return p
                        rel_p = settings.model_dir / local_path
                        if rel_p.exists():
                            return rel_p

        # 2. Known default path heuristics under settings.model_dir
        root = settings.model_dir
        path_map = {
            "satquery_buildings_bundle": root / "satquery_buildings_bundle",
            "satquery_water_bundle": root / "satquery_water_bundle",
            "water_finetuned": root / "satquery_water_bundle",
            "dinov3s-buildings": root / "buildings/dinov3s-buildings/onnx/model.onnx",
            "building_primary": root / "buildings/dinov3s-buildings/onnx/model.onnx",
            "hotosm/dinov3s-buildings": root / "buildings/dinov3s-buildings/onnx/model.onnx",
            "geobase/dinov3s-buildings": root / "buildings/dinov3s-buildings/onnx/model.onnx",
            "geobase/building-detection": root / "buildings/geobase-building-detection/onnx/model_quantized.onnx",
            "building_secondary": root / "buildings/geobase-building-detection/onnx/model_quantized.onnx",
            "flair-rgb": root / "land_cover/flair-rgb-15cl-deeplabv3/FLAIR-INC_rgb_15cl_resnet34-deeplabv3_weights.pth",
            "flair": root / "land_cover/flair-rgb-15cl-deeplabv3/FLAIR-INC_rgb_15cl_resnet34-deeplabv3_weights.pth",
            "land_rgb": root / "land_cover/flair-rgb-15cl-deeplabv3/FLAIR-INC_rgb_15cl_resnet34-deeplabv3_weights.pth",
            "bigearthnet-s2": root / "land_cover/bigearthnet-s2-resnet50/model.safetensors",
            "land_s2": root / "land_cover/bigearthnet-s2-resnet50/model.safetensors",
            "bigearthnet-s1": root / "land_cover/bigearthnet-s1-resnet50/model.safetensors",
            "land_s1": root / "land_cover/bigearthnet-s1-resnet50/model.safetensors",
            "bigearthnet-s1s2": root / "land_cover/bigearthnet-s1s2-resnet101/model.safetensors",
            "land_s1s2": root / "land_cover/bigearthnet-s1s2-resnet101/model.safetensors",
            "prithvi": root / "water/prithvi-sen1floods11/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
            "water_s2": root / "water/prithvi-sen1floods11/Prithvi-EO-V2-300M-TL-Sen1Floods11.pt",
            "earthdial": root / "vlm/earthdial-4b",
            "earthdial-4b-rgb": root / "vlm/earthdial-4b/EarthDial_4B_RGB",
            "earthdial-4b-ms": root / "vlm/earthdial-4b/EarthDial_4B_MS",
            "croma": root / "fusion/croma/CROMA_base.pt",
            "croma_base": root / "fusion/croma/CROMA_base.pt",
            "croma_large": root / "fusion/croma/CROMA_large.pt",
            "remoteclip": root / "remoteclip/rn50/RemoteCLIP-RN50.pt",
            "remoteclip_rn50": root / "remoteclip/rn50/RemoteCLIP-RN50.pt",
            "remoteclip_vit_b_32": root / "remoteclip/vit-b-32/RemoteCLIP-ViT-B-32.pt",
            "tinycd": root / "change_detection/tinycd/levir_best.pth",
            "bit": root / "change_detection/bit/bit_r18_256x256_40k_levircd.pth",
            "changerex": root / "change_detection/changerex/ChangerEx_r18-512x512_40k_levircd.pth",
            "ban": root / "change_detection/ban/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.pth",
        }
        for k, p in path_map.items():
            if clean_id in (k, k.replace("_", "-"), k.replace("-", "_")):
                if p.exists():
                    return p

        # 3. Fallback to existing legacy model files if present
        legacy_candidates = [
            root / "buildings/model.onnx",
            root / "croma/CROMA_base.pt",
            root / "remoteclip/RemoteCLIP-RN50.pt",
            root / "remoteclip/RemoteCLIP-ViT-B-32.pt",
            root / "earthdial/EarthDial_4B_RGB",
            root / "earthdial/EarthDial_4B_MS",
            root / "landcover/deeplabv3_landcover_4c.onnx",
        ]
        for c in legacy_candidates:
            if clean_id in str(c).lower() and c.exists():
                return c

        return None

    def __init__(self) -> None:
        self.health: dict[str, dict] = {}
        self.local_registry = LocalModelRegistry(settings.model_dir, self.get_manifest_path())
        self.specs = {
            "earthdial": ModelSpec(
                "earthdial",
                "Remote-sensing VQA, captioning, grounding and explanation",
                settings.earthdial_endpoint,
                ("query", "image_paths"),
            ),
            "earthdial-4b-rgb": ModelSpec(
                "earthdial-4b-rgb",
                "High-resolution 3-band Optical Remote-Sensing VQA & Grounding",
                settings.earthdial_rgb_endpoint,
                ("query", "image_paths"),
            ),
            "earthdial-4b-ms": ModelSpec(
                "earthdial-4b-ms",
                "Multispectral & multi-sensor remote-sensing VQA & Grounding",
                settings.earthdial_ms_endpoint,
                ("query", "image_paths"),
            ),
            "croma": ModelSpec(
                "croma",
                "Independent Sentinel-1/Sentinel-2 radar-optical representation evidence",
                settings.croma_endpoint,
                ("optical_path", "sar_path"),
            ),
            "remoteclip": ModelSpec(
                "remoteclip",
                "Text-to-tile semantic retrieval",
                settings.remoteclip_endpoint,
                ("query", "tile_paths"),
            ),
            "change": ModelSpec(
                "change",
                "External TinyCD/Open-CD change-mask inference",
                settings.change_endpoint,
                ("before_path", "after_path"),
            ),
            "vlm": ModelSpec(
                "vlm",
                "Provider-neutral LLaVA/BLIP-2/managed multimodal inference",
                settings.vlm_endpoint,
                ("query", "image_paths", "tile_paths", "metadata"),
            ),
        }

    def resolve_earthdial_variant(self, metadata: list[dict[str, Any]] | None = None) -> str:
        if not metadata:
            return "earthdial-4b-rgb"
        primary = metadata[0] if isinstance(metadata, list) and metadata else {}
        bands = primary.get("bands", 3)
        band_names = [str(n).lower() for n in primary.get("band_names", [])]
        if bands > 3 or any(k in band_names for k in ("nir", "swir", "rededge", "water_vapor")):
            return "earthdial-4b-ms"
        return "earthdial-4b-rgb"

    def capabilities(self) -> list[dict[str, Any]]:
        from importlib.util import find_spec
        import json
        metrics_path = Path(__file__).resolve().parents[3] / "artifacts/building_benchmark/benchmark.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
        local_models = self.local_registry.load()
        building_manifest = local_models.get("building_primary")
        building_ready = bool(building_manifest and building_manifest.availability)

        rf_config_path = Path(settings.building_checkpoint) / "model.safetensors"
        bundle_config_path = Path(settings.building_checkpoint) / "satquery_buildings_config.json"
        is_rf = Path(settings.building_checkpoint).is_dir() and rf_config_path.is_file()
        is_bundle = Path(settings.building_checkpoint).is_dir() and bundle_config_path.is_file()
        if is_rf:
            from satquery_engine.services.buildings_rf import MODEL_ID as rf_model_id, REVISION as rf_revision
            satellite_manifest = local_models.get("building_satellite")
            satellite_ready = bool(satellite_manifest and satellite_manifest.availability and
                                   satellite_manifest.local_path.resolve() == Path(settings.building_checkpoint).resolve())
            building_entry = {
                "name": "buildings", "model_id": rf_model_id,
                "purpose": "Satellite building instance segmentation and footprint counting",
                "available": satellite_ready, "status": "READY" if satellite_ready else "DEGRADED",
                "required_inputs": ["rgb_image"], "checkpoint_path": str(settings.building_checkpoint),
                "version": rf_revision, "accepted_modalities": ["optical"],
                "required_bands": ["red", "green", "blue"],
                "input_type": "RGB scene tiled at native pixels", "output_type": "building instance masks",
                "supported_tasks": ["BUILDING_COUNT", "BUILDING_FOOTPRINT", "BUILDING_CHANGE"],
                "expected_resolution": "very high resolution satellite RGB; validate for deployment",
                "hardware_requirements": "PyTorch CUDA or CPU", "calibration_version": None,
                "evaluation_metrics": None,
                "training_data": "merve/satellite-building-segmentation",
                "license_provenance": "https://huggingface.co/merve/rf-detr-seg-satellite-buildings",
                "failure_modes": ["aerial domain shift", "coarse resolution", "occlusion", "tiny roofs"],
            }
        elif is_bundle:
            bundle_metrics_path = Path(settings.building_checkpoint) / "satquery_buildings_metrics.json"
            bundle_metrics = json.loads(bundle_metrics_path.read_text()) if bundle_metrics_path.is_file() else None
            building_entry = {
                "name": "buildings",
                "model_id": "satquery_buildings_bundle",
                "purpose": "Fine-tuned building footprint segmentation and instance counting (SpaceNet 2)",
                "available": building_ready,
                "status": building_manifest.health_status.value if building_manifest else "READY",
                "required_inputs": ["rgb_image"],
                "checkpoint_path": str(settings.building_checkpoint),
                "version": "1.0.0",
                "accepted_modalities": ["optical"],
                "required_bands": ["red", "green", "blue"],
                "input_type": "RGB tile (512x512)",
                "output_type": "2-class logits (footprint, boundary)",
                "supported_tasks": ["BUILDING_COUNT", "BUILDING_FOOTPRINT", "BUILDING_CHANGE"],
                "expected_resolution": "very high resolution; validate for deployment",
                "hardware_requirements": "PyTorch CPU",
                "calibration_version": None,
                "evaluation_metrics": bundle_metrics or metrics,
                "training_data": "SpaceNet 2 (Vegas, Shanghai, Paris, Khartoum)",
                "license_provenance": "models/satquery_buildings_bundle",
                "failure_modes": ["dense touching roofs", "shadows", "coarse resolution", "off-nadir displacement"],
            }
        else:
            building_entry = {
                "name": "buildings", "model_id": "hotosm/dinov3s-buildings", "purpose": "Building footprint segmentation and instance counting",
                "available": building_ready, "status": building_manifest.health_status.value if building_manifest else "MISSING",
                "required_inputs": ["rgb_image"], "checkpoint_path": str(settings.building_checkpoint),
                "version": "f7254545dfbbf10a40594d5a5fb62136902000ae", "accepted_modalities": ["optical"],
                "required_bands": ["red", "green", "blue"], "input_type": "RGB tile", "output_type": "3-class logits",
                "supported_tasks": ["BUILDING_COUNT", "BUILDING_FOOTPRINT", "BUILDING_CHANGE"],
                "expected_resolution": "very high resolution; validate for deployment", "hardware_requirements": "ONNX Runtime CPU",
                "calibration_version": None, "evaluation_metrics": metrics, "training_data": "hotosm/vhr-building-segmentation",
                "license_provenance": "https://huggingface.co/hotosm/dinov3s-buildings",
                "failure_modes": ["touching buildings", "shadows", "coarse resolution", "domain shift"],
            }
        return [building_entry] + [
            {
                **{k:v for k,v in asdict(spec).items() if k != "endpoint"},
                "name": spec.name,
                "model_id": spec.name,
                "purpose": spec.purpose,
                "available": self.health.get(spec.name, {}).get("status") == "READY",
                "status": self.health.get(spec.name, {}).get("status", "DEGRADED" if spec.endpoint else "MISSING"),
                "required_inputs": list(spec.required_inputs),
            }
            for spec in self.specs.values()
        ] + [
            {
                **manifest.to_dict(),
                "name": key,
                "status": manifest.health_status.value,
                "available": manifest.availability,
                "checkpoint_path": str(manifest.local_path),
                "required_bands": list(manifest.expected_bands),
                "accepted_modalities": list(manifest.modality),
                "supported_tasks": list(manifest.task),
            }
            for key, manifest in local_models.items()
            if key != "building_primary"
        ]

    async def check_health(self):
        import asyncio
        local_models = await asyncio.to_thread(self.local_registry.load, refresh=True)
        aliases = {
            "earthdial": "earthdial", "earthdial-4b-rgb": "earthdial", "earthdial-4b-ms": "earthdial",
            "croma": "croma_base", "remoteclip": "remoteclip_rn50", "change": "tinycd",
        }
        for name, local_key in aliases.items():
            manifest = local_models.get(local_key)
            if manifest:
                self.health[name] = {
                    "status": manifest.health_status.value,
                    "inference_verified": manifest.availability,
                    "local_path": str(manifest.local_path),
                    **manifest.metadata,
                }

        if settings.offline_mode:
            for name in self.specs:
                self.health.setdefault(name, {"status": "MISSING", "inference_verified": False})
            return

        for name, spec in self.specs.items():
            if not spec.endpoint:
                self.health.setdefault(name, {"status": "MISSING", "inference_verified": False})
                continue
            try:
                async with httpx.AsyncClient(timeout=2) as client:
                    response = await client.get(spec.endpoint.rstrip('/') + '/health')
                    response.raise_for_status()
                    info = response.json()
                ready = info.get("inference_verified") is True and info.get("mock") is not True
                self.health[name] = {"status": "READY" if ready else "DEGRADED"}
            except Exception:
                self.health[name] = {"status": "LOAD_FAILED", "inference_verified": False}

    async def invoke(self, name: str, payload: dict[str, Any], timeout_seconds: float = 120) -> dict[str, Any]:
        spec = self.specs.get(name)
        if not spec or not spec.endpoint:
            env_var_name = f"SATQUERY_{name.upper().replace('-', '_')}_ENDPOINT"
            return {
                "available": False,
                "model": name,
                "reason": f"{env_var_name} is not configured",
            }

        if settings.offline_mode:
            return {
                "available": False,
                "model": name,
                "reason": f"SATQUERY_OFFLINE_MODE is enabled; external network calls for {name} are blocked.",
            }

        missing = [key for key in spec.required_inputs if key not in payload]
        if missing:
            raise ValueError(f"Missing inputs for {name}: {missing}")

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                health = await client.get(f"{spec.endpoint.rstrip('/')}/health", timeout=3)
                health.raise_for_status()
                identity = health.json()
                if identity.get("mock") or identity.get("inference_verified") is not True:
                    raise ValueError("Model service has not verified real inference; mock/unverified service rejected")
                response = await client.post(f"{spec.endpoint.rstrip('/')}/infer", json=payload)
                response.raise_for_status()
                result = response.json()
            if not isinstance(result, dict) or result.get("mock") or result.get("available") is False:
                raise ValueError("Model did not return a valid inference result")
            if "confidence" in result and not 0 <= float(result["confidence"]) <= 1:
                raise ValueError("Invalid model score")
            result["available"] = True
            result["model"] = name
            result["duration_ms"] = round((time.perf_counter() - started) * 1000)
            return result
        except Exception as err:
            return {
                "available": False,
                "model": name,
                "reason": f"External endpoint for {name} ({spec.endpoint}) unreachable: {err}",
            }

    async def invoke_earthdial_variant(
        self,
        *,
        variant: str,
        query: str,
        image_paths: list[str],
        tile_paths: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not image_paths:
            return {"available": False, "model": variant, "reason": "Real input images are required; no benchmark fallback is permitted."}
        return await self.invoke(variant, {"query": query, "image_paths": image_paths, "tile_paths": tile_paths, "metadata": metadata})

    async def invoke_vlm(
        self,
        *,
        query: str,
        image_paths: list[str],
        tile_paths: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if settings.vlm_endpoint and self.health.get("vlm", {}).get("status") == "READY":
            try:
                connector = VLMConnector(
                    endpoint=settings.vlm_endpoint,
                    api_key=settings.vlm_api_key,
                    model_name=settings.vlm_model_name,
                )
                return await connector.infer(
                    query=query,
                    image_paths=[Path(path) for path in image_paths],
                    tile_paths=[Path(path) for path in tile_paths],
                    metadata=metadata,
                )
            except Exception:
                pass
        variant = self.resolve_earthdial_variant(metadata)
        spec = self.specs.get(variant)
        if spec and spec.endpoint:
            return await self.invoke(variant, {"query": query, "image_paths": image_paths})
        return await self.invoke("earthdial", {"query": query, "image_paths": image_paths})


registry = ModelRegistry()
