import numpy as np
import pytest

from satquery.registry import InputContractError, ModelRegistry


def test_detects_rgb_and_sentinel2_only_with_exact_order():
    registry = ModelRegistry()
    assert registry.detect_sensor(3, []) == "RGB_VHR"
    expected = registry.model("water_sentinel2_v1").band_order
    assert registry.detect_sensor(13, expected) == "SENTINEL2_L1C_13B"
    with pytest.raises(InputContractError, match="explicit"):
        registry.detect_sensor(13, [])
    with pytest.raises(InputContractError, match="order mismatch"):
        registry.detect_sensor(13, list(reversed(expected)))


def test_rejects_wrong_scale_and_band_order():
    registry = ModelRegistry()
    spec = registry.model("landcover_rgb_v1")
    with pytest.raises(InputContractError, match="expects bands"):
        registry.validate_array(spec, np.zeros((3, 8, 8), "float32"), ["blue", "green", "red"])
    with pytest.raises(InputContractError, match="scale"):
        registry.validate_array(spec, np.full((3, 8, 8), 10000, "float32"), ["red", "green", "blue"])

