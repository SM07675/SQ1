from satquery_engine.config import settings
from satquery_engine.models.registry import LocalModelRegistry


def test_bigearthnet_is_registered_as_scene_evidence_not_dense_mask():
    registry=LocalModelRegistry(settings.model_dir)
    for key in ("land_s2","land_s1","land_s1s2"):
        manifest=registry.get(key)
        assert manifest is not None
        assert "multilabel" in manifest.output_type
        assert "segmentation" not in manifest.output_type


def test_flair_is_the_registered_rgb_dense_landcover_specialist():
    manifest=LocalModelRegistry(settings.model_dir).get("land_rgb")
    assert manifest is not None
    assert manifest.adapter == "FlairLandCoverAdapter"
    assert "pixel" in manifest.output_type
