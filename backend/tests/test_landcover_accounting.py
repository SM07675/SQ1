from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from app.services.spectral import classify_land_cover_scene
from app.services.raster import render_preview


def test_landcover_percentages_exclude_nodata_and_do_not_double_count(tmp_path: Path) -> None:
    source = tmp_path / "scene.tif"
    bands = np.zeros((3, 64, 64), dtype=np.uint8)
    rows, cols = np.mgrid[0:32, 0:32]
    bands[0, 16:48, 16:48] = 25 + rows // 4
    bands[1, 16:48, 16:48] = 90 + cols * 3
    bands[2, 16:48, 16:48] = 35 + rows // 4
    with rasterio.open(
        source, "w", driver="GTiff", width=64, height=64, count=3,
        dtype="uint8", nodata=0, crs="EPSG:32643",
        transform=from_origin(500000, 2000000, 10, 10),
    ) as dataset:
        dataset.write(bands)

    result = classify_land_cover_scene(source, tmp_path / "out", query="Classify land cover")

    assert result["valid_pixel_count"] == 32 * 32
    assert result["analysis_pixel_count"] == 64 * 64
    percentages = [part["percent"] for part in result["breakdown"].values()]
    assert abs(sum(percentages) - 100.0) <= 0.05
    assert result["vegetation_percent"] == round(
        result["breakdown"]["vegetation"]["percent"]
        + result["breakdown"]["forest"]["percent"], 2,
    )
    assert result["land_coverage_percent"] == 100.0
    with Image.open(result["mask_path"]) as mask:
        assert mask.getpixel((0, 0))[3] == 0


def test_tiff_preview_uses_rgb_band_names_and_nodata_alpha(tmp_path: Path) -> None:
    source = tmp_path / "sentinel.tif"
    bands = np.zeros((3, 20, 20), dtype=np.uint8)
    bands[:, 5:15, 5:15] = np.array([25, 80, 210], dtype=np.uint8)[:, None, None]
    with rasterio.open(
        source, "w", driver="GTiff", width=20, height=20, count=3,
        dtype="uint8", nodata=0,
    ) as dataset:
        dataset.write(bands)
        dataset.descriptions = ("B02", "B03", "B04")

    preview = render_preview(source, tmp_path / "preview.png")
    with Image.open(preview) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
        red, green, blue, alpha = image.getpixel((10, 10))
        assert red > green > blue
        assert alpha == 255
