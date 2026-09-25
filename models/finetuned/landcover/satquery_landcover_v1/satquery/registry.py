from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class InputContractError(ValueError):
    """Raised when an image does not match a model's declared sensor contract."""


def canonical_band(name: str) -> str:
    value = str(name).strip().upper().replace("-", "").replace("_", "")
    aliases = {"RED": "RED", "GREEN": "GREEN", "BLUE": "BLUE", "B8A": "B8A"}
    if value in aliases:
        return aliases[value]
    if value.startswith("B") and value[1:].isdigit():
        return f"B{int(value[1:]):02d}"
    return value


@dataclass(frozen=True)
class ModelSpec:
    key: str
    raw: dict[str, Any]
    root: Path

    @property
    def checkpoint(self) -> Path | None:
        value = self.raw.get("checkpoint")
        return (self.root / value).resolve() if value else None

    @property
    def band_order(self) -> tuple[str, ...]:
        return tuple(self.raw["band_order"])

    def require_available(self) -> None:
        checkpoint = self.checkpoint
        if not self.raw.get("available") or checkpoint is None:
            required = self.raw.get("training_data_provenance", {}).get("required", "Train and register a checkpoint.")
            raise RuntimeError(f"Model '{self.key}' is registered but unavailable. {required}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Registered checkpoint for '{self.key}' does not exist: {checkpoint}")
        expected = self.raw.get("checkpoint_sha256")
        if expected:
            digest = hashlib.sha256()
            with checkpoint.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            actual = digest.hexdigest()
            if actual.lower() != expected.lower():
                raise RuntimeError(f"Checkpoint integrity failure for '{self.key}': expected {expected}, got {actual}.")


class ModelRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or Path(__file__).resolve().parents[1] / "model_registry.json").resolve()
        self.root = self.path.parent
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if self.data.get("format_version") != 1:
            raise ValueError("Unsupported model registry format.")

    def model(self, key: str) -> ModelSpec:
        try:
            return ModelSpec(key, self.data["models"][key], self.root)
        except KeyError as exc:
            raise KeyError(f"Unknown model registry key: {key}") from exc

    def detect_sensor(self, band_count: int, band_names: Iterable[str] | None) -> str:
        names = tuple(canonical_band(x) for x in (band_names or ()))
        if band_count == 3:
            if names and set(names) != {"RED", "GREEN", "BLUE"}:
                raise InputContractError(f"Three-band input must be explicitly RGB; received {names}.")
            return "RGB_VHR"
        if band_count == 13:
            expected = tuple(canonical_band(x) for x in self.model("water_sentinel2_v1").band_order)
            if not names:
                raise InputContractError("A 13-band raster needs explicit Sentinel-2 band descriptions or --band-order; positional guessing is disabled.")
            if names != expected:
                raise InputContractError(f"Sentinel-2 band order mismatch. Expected {expected}; received {names}.")
            return "SENTINEL2_L1C_13B"
        raise InputContractError(f"Unsupported input with {band_count} bands. Expected RGB (3) or Sentinel-2 L1C (13).")

    def validate_array(self, spec: ModelSpec, array: np.ndarray, band_names: Iterable[str]) -> None:
        if array.ndim != 3 or array.shape[0] != len(spec.band_order):
            raise InputContractError(f"Model '{spec.key}' requires {len(spec.band_order)} bands, got shape {array.shape}.")
        received = tuple(canonical_band(x) for x in band_names)
        expected = tuple(canonical_band(x) for x in spec.band_order)
        if received != expected:
            raise InputContractError(f"Model '{spec.key}' expects bands {expected}; received {received}.")
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            raise InputContractError("Input has no finite pixels.")
        lo, hi = float(np.percentile(finite, 0.1)), float(np.percentile(finite, 99.9))
        allowed = spec.raw.get("accepted_input_ranges", [])
        if not any(lo >= bounds[0] - 1e-6 and hi <= bounds[1] * 1.10 + 1e-6 for bounds in allowed):
            raise InputContractError(f"Input scale ({lo:.3g}..{hi:.3g}) is incompatible with '{spec.key}' ranges {allowed}.")
