from satquery_engine.models.adapters.base import (
    BigEarthNetAdapter,
    BigEarthNetFusionAdapter,
    BigEarthNetS1Adapter,
    BigEarthNetS2Adapter,
    ChangeDetectionAdapter,
    ChangeModelAdapter,
    CromaAdapter,
    DinoBuildingAdapter,
    EarthDialAdapter,
    FlairLandCoverAdapter,
    GenericModelAdapter,
    ModelAdapter,
    PrithviWaterAdapter,
    RemoteClipAdapter,
    SatlasBuildingAdapter,
    SatlasWaterAdapter,
    WaterShadowAdapter,
    adapter_for,
)

__all__ = [name for name in globals() if name.endswith("Adapter") or name == "adapter_for"]
