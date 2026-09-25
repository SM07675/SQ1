from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from satquery_engine.models.manifest import HealthStatus, ModelManifest


def _canon(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    status: HealthStatus
    reasons: tuple[str, ...] = ()
    selected_channels: tuple[int, ...] = ()
    score: float = 0.0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "compatible": self.compatible,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "selected_channels": list(self.selected_channels),
            "score": self.score,
            "warnings": list(self.warnings),
        }


class ModelCompatibility:
    """Fail-closed compatibility validation before an adapter or model is called."""

    @staticmethod
    def evaluate(manifest: ModelManifest, raster_profile: Any, task: str | None = None) -> CompatibilityResult:
        reasons: list[str] = []
        warnings: list[str] = []
        if manifest.health_status != HealthStatus.READY:
            reasons.append(f"model health is {manifest.health_status.value}, not READY")
        if task and task not in manifest.task:
            reasons.append(f"task {task!r} is not supported")

        modality = _canon(str(getattr(raster_profile, "modality", "unknown")))
        accepted = {_canon(item) for item in manifest.modality}
        aliases = {
            "optical": {"optical", "rgb"},
            "rgb": {"rgb", "optical"},
            "multispectral": {"multispectral", "multispectral_s2"},
            "sar": {"sar", "sar_s1"},
        }
        if accepted and not any(modality in aliases.get(item, {item}) or item in aliases.get(modality, {modality}) for item in accepted):
            reasons.append(f"input modality {modality!r} is incompatible with {sorted(accepted)}")

        band_map = getattr(raster_profile, "band_map", {}) or {}
        indices = band_map.get("indices", band_map) if isinstance(band_map, dict) else {}
        available = {_canon(str(name)): int(index) for name, index in indices.items() if isinstance(index, int)}
        selected: list[int] = []
        count = int(getattr(raster_profile, "bands", getattr(raster_profile, "band_count", 0)) or 0)
        ordinary_rgb = (
            count == 3
            and {_canon(item) for item in manifest.expected_band_order} == {"red", "green", "blue"}
            and modality in {"rgb", "optical"}
            and not available
        )
        if ordinary_rgb:
            available = {"red":1,"green":2,"blue":3}
            warnings.append("Using the documented ordinary three-channel RGB convention.")
        for band in manifest.expected_band_order:
            key = _canon(band)
            band_aliases = {
                "b02": "blue", "b03": "green", "b04": "red", "b08": "nir",
                "b8a": "narrow_nir", "narrow_nir": "narrow_nir", "b11": "swir1", "b12": "swir2",
                "sar_vh": "vh", "sar_vv": "vv",
            }
            key = band_aliases.get(key, key)
            if key == "narrow_nir" and key not in available and "nir" in available:
                warnings.append("Narrow NIR was approximated by a generic NIR band; exact checkpoint compatibility is not established.")
                reasons.append("exact Narrow NIR band is missing")
                continue
            if key not in available:
                reasons.append(f"required band {band} is missing or untrusted")
            else:
                selected.append(available[key])

        if manifest.input_channels and len(manifest.expected_band_order) != manifest.input_channels:
            reasons.append("manifest channel contract is internally inconsistent")
        if not manifest.expected_band_order and manifest.input_channels and count < manifest.input_channels:
            reasons.append(f"input has {count} channels; {manifest.input_channels} required")

        resolution = getattr(raster_profile, "resolution", None)
        if resolution and "0.2" in manifest.expected_resolution:
            gsd = max(abs(float(resolution[0])), abs(float(resolution[1])))
            if gsd > 0.75:
                warnings.append(f"{gsd:g} map-units/pixel is outside the documented VHR model domain.")

        compatible = not reasons
        score = 1.0 if compatible else max(0.0, 1.0 - 0.25 * len(reasons) - 0.05 * len(warnings))
        return CompatibilityResult(
            compatible=compatible,
            status=HealthStatus.READY if compatible else HealthStatus.INCOMPATIBLE_INPUT,
            reasons=tuple(reasons),
            selected_channels=tuple(selected),
            score=round(score, 3),
            warnings=tuple(warnings),
        )
