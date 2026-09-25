"""Model contracts, compatibility checks, adapters, and lazy runtime resources."""

from satquery_engine.models.compatibility import CompatibilityResult, ModelCompatibility
from satquery_engine.models.manifest import HealthStatus, ModelManifest
from satquery_engine.models.runtime import ModelRuntimeManager
from satquery_engine.models.registry import LocalModelRegistry, ModelHealthCheck
from satquery_engine.models.adapters import ModelAdapter

__all__ = [
    "CompatibilityResult",
    "HealthStatus",
    "ModelCompatibility",
    "ModelManifest",
    "ModelAdapter",
    "ModelHealthCheck",
    "LocalModelRegistry",
    "ModelRuntimeManager",
]
