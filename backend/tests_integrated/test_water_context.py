import numpy as np

from satquery_engine.services.water_engine import compute_rgb_water_proxy, postprocess_water_mask


def test_dark_connected_water_recovers_without_isolated_shadow():
    rgb = np.full((3, 100, 140), .7, dtype=np.float32)
    rgb[:, 10:90, 10:40] = np.array([.025, .055, .09])[:, None, None]
    rgb[:, 10:90, 40:85] = .025
    rgb[:, 25:75, 105:130] = .025  # separate neutral building shadow
    valid = np.ones((100, 140), bool)
    prob, seeds, _, _ = compute_rgb_water_proxy(*rgb, valid)
    mask, _, _ = postprocess_water_mask(prob, valid, seeds, filter_compact=False)
    # Edge-density support is deliberately conservative at the bright bank.
    assert mask[20:80, 45:75].all()
    assert mask[20:80, 45:80].mean() > .90
    assert not mask[30:70, 110:125].any()


def test_closing_does_not_fill_rejected_cloud_or_land():
    prob = np.full((40, 40), .8, dtype=np.float32)
    prob[20, 20] = .05
    valid = np.ones(prob.shape, bool)
    mask, _, _ = postprocess_water_mask(prob, valid, filter_compact=False)
    assert not mask[20, 20]
    assert mask[19, 20]


def test_water_reconstruction_cannot_promote_rejected_land_seeds():
    prob = np.full((40, 40), .1, dtype=np.float32)
    seeds = np.ones_like(prob, dtype=bool)
    mask, _, _ = postprocess_water_mask(prob, seeds, seeds, filter_compact=False)
    assert not mask.any()


def test_dark_recovery_never_crosses_invalid_pixels():
    rgb = np.full((3, 50, 80), .025, dtype=np.float32)
    rgb[:, :, :20] = np.array([.025, .055, .09])[:, None, None]
    valid = np.ones((50, 80), bool)
    valid[:, 22:25] = False
    prob, seeds, _, _ = compute_rgb_water_proxy(*rgb, valid)
    mask, _, _ = postprocess_water_mask(prob, valid, seeds, filter_compact=False)
    assert not mask[:, 22:].any()


def test_supported_five_band_aerial_rgb_is_not_rejected(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    from satquery_engine.services.water_engine import execute_water_pipeline

    path = tmp_path / "aerial.tif"
    data = np.full((5, 40, 40), 150, dtype="uint8")
    data[:3, :, :] = np.array([20, 50, 90], dtype="uint8")[:, None, None]
    with rasterio.open(path, "w", driver="GTiff", width=40, height=40, count=5,
                       dtype="uint8", crs="EPSG:2154", transform=from_origin(700000, 6600000, .2, .2)) as dst:
        dst.write(data)
    result = execute_water_pipeline(path, tmp_path / "result", use_model=False)
    assert result["route"] == "RGB_WATER_PROXY"
    assert result["selected_pixels"] > 1400
