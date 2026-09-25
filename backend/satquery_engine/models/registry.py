from __future__ import annotations

import importlib.util
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import yaml

from satquery_engine.models.adapters import adapter_for
from satquery_engine.models.manifest import HealthStatus, ModelManifest


_CONTRACTS: dict[str, dict[str, Any]] = {
    "water_s2_surface": dict(adapter="GenericModelAdapter", required_files=("model.pth", "config.json"), required_packages=("torch", "segmentation_models_pytorch"),
        expected_resolution="Named six-band Sentinel-2, 8-30 m", normalization={"type":"scale", "factor":1.0/255.0}, input_dtype="float32", input_range="publisher-native Sentinel-2 DN"),
    "land_deepness": dict(adapter="GenericModelAdapter", required_files=("deeplabv3_landcover_4c.onnx",), required_packages=("onnxruntime",),
        expected_resolution="0.15-0.60 m aerial RGB", normalization={"type":"scale", "factor":1.0/255.0}, input_dtype="float32", input_range="RGB 0..255"),
    "building_satellite": dict(adapter="GenericModelAdapter", required_files=("model.safetensors", "config.json", "preprocessor_config.json"), required_packages=("torch", "transformers"),
        expected_resolution="VHR satellite RGB, <=1.5 m when known", normalization={"type":"checkpoint_native", "processor":"RfDetrImageProcessor"}, input_dtype="float32", input_range="RGB 0..255; production processor rescales and normalizes"),
    "satlas_aerial_swinb_si": dict(adapter="SatlasBuildingAdapter", required_files=("aerial_swinb_si.pth", "satlas_metadata.json"), required_packages=("torch", "satlaspretrain_models"),
        expected_resolution="0.5-2 m aerial RGB", normalization={"type":"scale", "factor":1.0/255.0}, input_dtype="float32", input_range="0..255 RGB"),
    "land_flair_hub": dict(adapter="FlairLandCoverAdapter", required_files=("FLAIR-HUB_LC-A_RGB_swinbase-upernet.safetensors", "configs_train/config_models.yaml", "configs_train/config_modalities.yaml", "configs_train/config_supervision.yaml"), required_packages=("torch", "segmentation_models_pytorch", "safetensors"),
        expected_resolution="FLAIR-HUB high-resolution aerial RGB", normalization={"type":"mean_std", "mean":[105.66,111.35,102.18], "std":[52.23,45.62,44.30]}, input_dtype="float32", input_range="0..255 RGB"),
    "water_shadow_rgb": dict(adapter="WaterShadowAdapter", required_files=("satquery_water_shadow_config.json", "satquery_water_shadow_state_dict.pt"), required_packages=("torch",),
        expected_resolution="RGB VHR", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="checkpoint-native RGB"),
    "landcover_primary": dict(adapter="FlairLandCoverAdapter", required_files=("best_checkpoint.pt", "class_map.json"), required_packages=("torch", "segmentation_models_pytorch"),
        expected_resolution="RGB VHR", normalization={"type":"scale", "factor":1.0/255.0}, input_dtype="float32", input_range="0..1 RGB"),
    "building_fallback": dict(adapter="DinoBuildingAdapter", required_files=("onnx/model.onnx", "model.ckpt", "README.md"), required_packages=("onnxruntime",),
        expected_resolution="VHR aerial or satellite RGB", normalization={"type":"mean_std", "mean":[0.4296737853,0.4001659668,0.3433337280], "std":[0.2056069389,0.1673855556,0.1598986423]}, input_dtype="float32", input_range="0..1 RGB"),
    "building_primary": dict(adapter="DinoBuildingAdapter", required_files=("onnx/model.onnx",), required_packages=("onnxruntime",),
        expected_resolution="0.2-0.75 m VHR RGB", normalization={"type":"mean_std","mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]}, input_dtype="float32", input_range="0..1 reflectance/RGB"),
    "building_secondary": dict(adapter="DinoBuildingAdapter", required_files=("onnx/model_quantized.onnx",), required_packages=("onnxruntime",),
        expected_resolution="VHR RGB", normalization={"type":"mean_std","mean":[0.485,0.456,0.406],"std":[0.229,0.224,0.225]}, input_dtype="float32", input_range="0..1 RGB"),
    "water_finetuned": dict(
        adapter="SatlasWaterAdapter",
        required_files=("satquery_water_config.json", "satquery_water_state_dict.pt", "thresholds.json"),
        required_packages=("torch", "torchvision"),
        expected_resolution="Sentinel-2 10/20 m on a co-registered grid",
        normalization={
            "type": "bundle_contract",
            "rgb": "clip(raw/3000,0,1)",
            "red_edge_nir_swir": "clip(raw/8160,0,1)",
            "source_scale": 10000,
        },
        input_dtype="float32",
        input_range="Sentinel-2 digital numbers (nominally 0..10000)",
    ),
    "land_rgb": dict(adapter="FlairLandCoverAdapter", required_files=("FLAIR-INC_rgb_15cl_resnet34-deeplabv3_weights.pth",), required_packages=("torch","segmentation_models_pytorch"),
        expected_resolution="0.2-0.5 m aerial RGB", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="checkpoint-native RGB"),
    "land_s2": dict(adapter="BigEarthNetS2Adapter", required_files=("model.safetensors","config.json"), required_packages=("torch","transformers"),
        expected_resolution="Sentinel-2 10/20 m", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="Sentinel-2 reflectance"),
    "land_s1": dict(adapter="BigEarthNetS1Adapter", required_files=("model.safetensors","config.json"), required_packages=("torch","transformers"),
        expected_resolution="Sentinel-1 GRD", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="calibrated SAR"),
    "land_s1s2": dict(adapter="BigEarthNetFusionAdapter", required_files=("model.safetensors","config.json"), required_packages=("torch","transformers"),
        expected_resolution="co-registered Sentinel-1 and Sentinel-2", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="calibrated S1/S2"),
    "water_s2": dict(adapter="PrithviWaterAdapter", required_files=("Prithvi-EO-V2-300M-TL-Sen1Floods11.pt","config.yaml"), required_packages=("torch","terratorch","transformers"),
        expected_resolution="Sentinel-2 10/20 m", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="Sentinel-2 reflectance"),
    "earthdial": dict(adapter="EarthDialAdapter", required_files=("model.safetensors.index.json","config.json","tokenizer.model"), required_packages=("torch","transformers"),
        expected_resolution="rendered RGB or documented multispectral variant", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="display RGB only for RGB variant"),
    "remoteclip_rn50": dict(adapter="RemoteClipAdapter", required_files=("RemoteCLIP-RN50.pt",), required_packages=("torch","open_clip"),
        expected_resolution="rendered remote-sensing RGB", normalization={"type":"mean_std","mean":[0.48145466,0.4578275,0.40821073],"std":[0.26862954,0.26130258,0.27577711]}, input_dtype="float32", input_range="0..1 display RGB"),
    "remoteclip_vit_b_32": dict(adapter="RemoteClipAdapter", required_files=("RemoteCLIP-ViT-B-32.pt",), required_packages=("torch","open_clip"),
        expected_resolution="rendered remote-sensing RGB", normalization={"type":"mean_std","mean":[0.48145466,0.4578275,0.40821073],"std":[0.26862954,0.26130258,0.27577711]}, input_dtype="float32", input_range="0..1 display RGB"),
    "croma_base": dict(adapter="CromaAdapter", required_files=("CROMA_base.pt",), required_packages=("torch","croma"),
        expected_resolution="co-registered Sentinel-1/2", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="calibrated S1/S2"),
    "tinycd": dict(adapter="ChangeDetectionAdapter", required_files=("levir_best.pth",), required_packages=("torch",),
        expected_resolution="registered VHR RGB temporal pair", normalization={"type":"scale","factor":1.0}, input_dtype="float32", input_range="0..1 RGB"),
    "bit": dict(adapter="ChangeDetectionAdapter", required_files=("bit_r18_256x256_40k_levircd.pth",), required_packages=("torch","mmengine","mmseg"),
        expected_resolution="registered VHR RGB temporal pair", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="0..1 RGB"),
    "changerex": dict(adapter="ChangeDetectionAdapter", required_files=("ChangerEx_r18-512x512_40k_levircd.pth",), required_packages=("torch","mmengine","mmseg"),
        expected_resolution="registered VHR RGB temporal pair", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="0..1 RGB"),
    "ban": dict(adapter="ChangeDetectionAdapter", required_files=("ban_vit-l14-clip_mit-b0_512x512_40k_levircd.pth",), required_packages=("torch","mmengine","mmseg"),
        expected_resolution="registered VHR RGB temporal pair", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="0..1 RGB"),
    "changemamba": dict(adapter="ChangeDetectionAdapter", required_files=("changemamba*.pth",), required_packages=("torch","selective_scan"),
        expected_resolution="registered VHR RGB temporal pair", normalization={"type":"checkpoint_native"}, input_dtype="float32", input_range="0..1 RGB"),
}


class ModelHealthCheck:
    """File, adapter, dependency, and lightweight checkpoint validation."""

    @staticmethod
    def _resolve_required(base: Path, pattern: str) -> Path | None:
        direct = base / pattern
        if direct.is_file():
            return direct
        matches = [p for p in base.glob(pattern) if p.is_file()]
        return matches[0] if matches else None

    @staticmethod
    def _looks_like_lfs_pointer(path: Path) -> bool:
        if path.stat().st_size > 1024:
            return False
        return path.read_bytes()[:100].startswith(b"version https://git-lfs.github.com/spec/v1")

    @staticmethod
    def _check_safetensors(path: Path) -> None:
        with path.open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
            if header_len <= 2 or header_len > path.stat().st_size - 8:
                raise ValueError("invalid safetensors header length")
            json.loads(stream.read(header_len))

    def check(self, manifest: ModelManifest) -> ModelManifest:
        if not manifest.enabled:
            return manifest.with_health(HealthStatus.DISABLED, False, reason="disabled by manifest")
        if not manifest.local_path.exists():
            return manifest.with_health(HealthStatus.MISSING, False, reason="model path does not exist")
        if not manifest.required_files:
            return manifest.with_health(HealthStatus.DEGRADED, False, reason="no validated checkpoint file contract is registered")
        missing: list[str] = []
        files: list[Path] = []
        for pattern in manifest.required_files:
            resolved = self._resolve_required(manifest.local_path, pattern) if manifest.local_path.is_dir() else manifest.local_path
            if resolved is None:
                missing.append(pattern)
            else:
                files.append(resolved)
        if missing:
            return manifest.with_health(HealthStatus.MISSING, False, missing_files=missing)
        try:
            adapter_for(manifest)
            for path in files:
                if not path.stat().st_size or self._looks_like_lfs_pointer(path):
                    raise ValueError(f"{path.name} is empty or an unresolved Git LFS pointer")
                if path.suffix == ".safetensors":
                    self._check_safetensors(path)
                elif path.suffix == ".onnx":
                    import onnxruntime as ort
                    # Avoid each startup graph allocating a full-machine thread
                    # pool while model inference and API work share this host.
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 4
                    options.inter_op_num_threads = 1
                    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                    if not session.get_inputs() or not session.get_outputs():
                        raise ValueError("ONNX graph has no inputs or outputs")
                    if manifest.metadata.get("registry_key") == "building_fallback":
                        # Health discovery must not execute an unused CPU model.
                        # Actual inference validates finite outputs in buildings.py.
                        expected = [1, 3, 256, 256]
                        if session.get_inputs()[0].shape != expected or session.get_outputs()[0].shape != expected:
                            raise ValueError("DINO fallback graph has an invalid schema")
                elif path.name.endswith("index.json"):
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    shards = set(payload.get("weight_map", {}).values())
                    if not shards or any(not (path.parent / shard).is_file() for shard in shards):
                        raise ValueError("checkpoint index references missing shards")
                elif path.suffix == ".json":
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if path.name == "satquery_buildings_config.json" and payload.get("architecture") != "SatlasBuildingNet":
                        raise ValueError("building bundle architecture is incompatible")
                    if path.name == "satquery_water_config.json" and payload.get("architecture") != "SatlasWaterNet":
                        raise ValueError("water bundle architecture is incompatible")
                elif path.suffix in {".pt", ".pth", ".ckpt"}:
                    with path.open("rb") as stream:
                        if len(stream.read(16)) < 16:
                            raise ValueError("checkpoint header is truncated")
                    if path.name in {"satquery_buildings_state_dict.pt", "satquery_water_state_dict.pt"}:
                        if importlib.util.find_spec("torch") is None:
                            continue
                        import torch
                        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                        if not isinstance(state, dict):
                            raise ValueError("fine-tuned checkpoint is not a state dictionary")
                        patch_key = "backbone.backbone.backbone.features.0.0.weight"
                        head_key = "head.weight"
                        if patch_key not in state or head_key not in state:
                            raise ValueError("fine-tuned checkpoint is missing required model layers")
                        expected_channels = 3 if "buildings" in path.name else 9
                        if tuple(state[patch_key].shape) != (128, expected_channels, 4, 4):
                            raise ValueError("fine-tuned checkpoint input layer does not match its bundle contract")
                        if tuple(state[head_key].shape[:2]) not in {(2, 64), (2, 96)}:
                            raise ValueError("fine-tuned checkpoint output head does not match its bundle contract")
                        if "water" in path.name:
                            spectral_key = "spectral.net.0.weight"
                            if spectral_key not in state or tuple(state[spectral_key].shape) != (32, 6, 3, 3):
                                raise ValueError("water checkpoint spectral branch is incompatible")
                        del state
        except Exception as exc:
            return manifest.with_health(HealthStatus.CORRUPT, False, reason=str(exc))

        missing_packages = [name for name in manifest.required_packages if importlib.util.find_spec(name) is None]
        if missing_packages:
            return manifest.with_health(
                HealthStatus.DEGRADED, False, reason="runtime dependencies are missing", missing_packages=missing_packages,
                files_verified=[str(p) for p in files], preprocessing_verified=True,
            )
        key = manifest.metadata.get("registry_key")
        if key == "building_satellite":
            from satquery_engine.services.buildings_rf import MODEL_SHA256
            with (manifest.local_path / "model.safetensors").open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != MODEL_SHA256:
                return manifest.with_health(HealthStatus.CORRUPT, False, reason="RF-DETR checkpoint identity mismatch")
        if key in {"land_s2", "land_s1", "land_s1s2"}:
            return manifest.with_health(HealthStatus.DEGRADED, False,
                reason="checkpoint structure verified; ConfigILM runtime and compatible sensor inference are not integrated")
        if key in {"satlas_aerial_swinb_si", "land_flair_hub"} or (key in {"building_primary", "water_finetuned"} and manifest.adapter in {"SatlasBuildingAdapter", "SatlasWaterAdapter"}):
            proof_path = manifest.local_path / "satquery_model_health.json"
            checkpoint = next((p for p in files if p.suffix in {".pth", ".pt", ".safetensors"}), None)
            if checkpoint is None or not proof_path.is_file():
                return manifest.with_health(HealthStatus.DEGRADED, False, reason="smoke inference has not been verified")
            try:
                proof = json.loads(proof_path.read_text(encoding="utf-8"))
                with checkpoint.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if proof.get("checkpoint_sha256") != digest or proof.get("inference_verified") is not True:
                    raise ValueError("health proof does not match the checkpoint")
                if key == "land_flair_hub" and proof.get("output_shape") != [1, 19, 512, 512]:
                    raise ValueError("FLAIR-HUB smoke output has the wrong class or spatial schema")
                if key == "satlas_aerial_swinb_si" and proof.get("checkpoint_id") != "Aerial_SwinB_SI":
                    raise ValueError("Satlas checkpoint ID mismatch")
                if key == "building_primary" and proof.get("output_shape") != [1, 2, 512, 512]:
                    raise ValueError("building task-head schema mismatch")
                if key == "water_finetuned" and proof.get("output_shape") != [1, 2, 512, 512]:
                    raise ValueError("Sentinel-2 water task-head schema mismatch")
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                return manifest.with_health(HealthStatus.DEGRADED, False, reason=str(exc))
        return manifest.with_health(
            HealthStatus.READY, True, files_verified=[str(p) for p in files], preprocessing_verified=True,
            smoke_test="graph_load" if any(p.suffix == ".onnx" for p in files) else "checkpoint_structure",
        )


_PROCESS_CACHE: dict[Path, dict[str, ModelManifest]] = {}


class LocalModelRegistry:
    def __init__(self, model_root: Path, manifest_path: Path | None = None) -> None:
        self.model_root = model_root.resolve()
        self.manifest_path = manifest_path or self.model_root / "manifests" / "models.yaml"
        self._manifests: dict[str, ModelManifest] | None = None

    def load(self, refresh: bool = False) -> dict[str, ModelManifest]:
        if not refresh and self._manifests is not None:
            return self._manifests
        if not refresh and self.manifest_path in _PROCESS_CACHE:
            self._manifests = _PROCESS_CACHE[self.manifest_path]
            return self._manifests
        if not self.manifest_path.is_file():
            self._manifests = {}
            return self._manifests
        payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8")) or {}
        entries = payload.get("models", payload)
        manifests: dict[str, ModelManifest] = {}
        for key, raw in entries.items():
            if not isinstance(raw, dict):
                continue
            contract = _CONTRACTS.get(key, {})
            raw_path = Path(str(raw.get("path", key)))
            candidate_path = raw_path if raw_path.is_absolute() else self.model_root / raw_path
            if (candidate_path.is_dir() and (candidate_path / "satquery_buildings_config.json").is_file()) or "satquery_buildings_bundle" in str(raw_path):
                contract = dict(
                    adapter="SatlasBuildingAdapter",
                    required_files=("satquery_buildings_config.json", "satquery_buildings_state_dict.pt", "thresholds.json"),
                    required_packages=("torch",),
                    expected_resolution="0.2-0.75 m VHR RGB",
                    normalization={"type": "scale", "factor": 1.0 / 255.0},
                    input_dtype="float32",
                    input_range="0..1 RGB",
                )
            if key != "water_shadow_rgb" and (
                (candidate_path.is_dir() and (candidate_path / "satquery_water_config.json").is_file())
                or "satquery_water_bundle" in str(raw_path)
            ):
                contract = dict(_CONTRACTS["water_finetuned"])
            if (candidate_path.is_dir() and (candidate_path / "best_checkpoint.pt").is_file()) or "satquery_landcover_v1" in str(raw_path):
                contract = dict(
                    adapter="FlairLandCoverAdapter",
                    required_files=("best_checkpoint.pt", "class_map.json"),
                    required_packages=("torch", "segmentation_models_pytorch"),
                    expected_resolution="RGB VHR",
                    normalization={"type": "scale", "factor": 1.0 / 255.0},
                    input_dtype="float32",
                    input_range="0..1 RGB",
                )
            if raw_path.is_absolute():
                try:
                    raw_path.resolve().relative_to(self.model_root)
                    local_path = raw_path.resolve()
                except ValueError:
                    local_path = self.model_root / key
            else:
                local_path = self.model_root / raw_path
            tasks = raw.get("task", ())
            if isinstance(tasks, str): tasks = (tasks,)
            modalities = raw.get("modality", ())
            if isinstance(modalities, str): modalities = (modalities,)
            bands = tuple(str(item) for item in raw.get("expected_channels", ()))
            input_size = tuple(int(item) for item in raw.get("input_size", ()) if isinstance(item, (int, float)))
            input_channels = len(bands) or (input_size[-1] if len(input_size) == 3 else 0)
            manifests[key] = ModelManifest(
                model_id=str(raw.get("model_id", key)), local_path=local_path, task=tuple(tasks), modality=tuple(modalities),
                expected_bands=bands, expected_band_order=bands, input_channels=input_channels,
                expected_resolution=str(contract.get("expected_resolution", raw.get("notes", {}).get("compatibility", "documented checkpoint domain"))),
                input_dtype=str(contract.get("input_dtype", "float32")), input_range=str(contract.get("input_range", "checkpoint-native")),
                normalization=dict(contract.get("normalization", {"type":"checkpoint_native"})), input_size=input_size,
                output_type=str(raw.get("output_type", "provider_output")), device="auto", precision="float32",
                adapter=str(contract.get("adapter", "GenericModelAdapter")), required_files=tuple(contract.get("required_files", ())),
                required_packages=tuple(contract.get("required_packages", ())), metadata={
                    "registry_key": key,
                    "priority": raw.get("priority"),
                    "display_name": raw.get("display_name", raw.get("model_id", key)),
                    "base_model": raw.get("base_model"),
                    "version": raw.get("version"),
                    "training_dataset": raw.get("training_dataset"),
                    "threshold_file": raw.get("threshold_file"),
                    "policy": raw.get("policy", raw.get("priority")),
                },
            )
        checker = ModelHealthCheck()
        self._manifests = {key: checker.check(manifest) for key, manifest in manifests.items()}
        _PROCESS_CACHE[self.manifest_path] = self._manifests
        return self._manifests

    def get(self, key_or_id: str) -> ModelManifest | None:
        target = key_or_id.lower()
        for key, manifest in self.load().items():
            if target in {key.lower(), manifest.model_id.lower(), manifest.model_id.rsplit("/", 1)[-1].lower()}:
                return manifest
        return None

    def dashboard(self) -> list[dict[str, Any]]:
        return [manifest.to_dict() for manifest in self.load().values()]
