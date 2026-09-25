from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class HealthStatus(str, Enum):
    READY = "READY"
    MISSING = "MISSING"
    CORRUPT = "CORRUPT"
    INCOMPATIBLE_INPUT = "INCOMPATIBLE_INPUT"
    LOAD_FAILED = "LOAD_FAILED"
    DEGRADED = "DEGRADED"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class ModelManifest:
    model_id: str
    local_path: Path
    task: tuple[str, ...]
    modality: tuple[str, ...]
    expected_bands: tuple[str, ...]
    expected_band_order: tuple[str, ...]
    input_channels: int
    expected_resolution: str
    input_dtype: str
    input_range: str
    normalization: dict[str, Any]
    input_size: tuple[int, ...]
    output_type: str
    device: str = "auto"
    precision: str = "float32"
    availability: bool = False
    health_status: HealthStatus = HealthStatus.MISSING
    adapter: str = "GenericModelAdapter"
    required_files: tuple[str, ...] = ()
    required_packages: tuple[str, ...] = ()
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["local_path"] = str(self.local_path)
        payload["health_status"] = self.health_status.value
        return payload

    def with_health(self, status: HealthStatus, available: bool, **metadata: Any) -> "ModelManifest":
        values = asdict(self)
        values["local_path"] = self.local_path
        values["health_status"] = status
        values["availability"] = available
        values["metadata"] = {**self.metadata, **metadata}
        return ModelManifest(**values)
