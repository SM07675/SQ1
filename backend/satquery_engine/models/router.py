from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from satquery_engine.models.compatibility import CompatibilityResult, ModelCompatibility
from satquery_engine.models.manifest import ModelManifest


@dataclass(frozen=True)
class RoutedModel:
    role: str
    manifest: ModelManifest
    compatibility: CompatibilityResult


class CompatibilityRouter:
    """Select the first healthy, input-compatible model for a task role."""

    def route(
        self,
        candidates: Iterable[tuple[str, ModelManifest]],
        raster_profile: Any,
        task: str,
    ) -> RoutedModel | None:
        for role, manifest in candidates:
            result = ModelCompatibility.evaluate(manifest, raster_profile, task)
            if result.compatible:
                return RoutedModel(role=role, manifest=manifest, compatibility=result)
        return None
