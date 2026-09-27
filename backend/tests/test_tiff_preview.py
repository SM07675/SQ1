import io
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _make_dummy_geotiff() -> bytes:
    buf = io.BytesIO()
    # 3-band 64x64 uint8 raster
    data = np.random.randint(20, 200, size=(3, 64, 64), dtype=np.uint8)
    transform = from_origin(100.0, 50.0, 10.0, 10.0)
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)
    buf.seek(0)
    return buf.getvalue()


def test_tiff_upload_generates_preview():
    # 1. Create chat
    create_res = client.post("/api/v1/chats", json={"title": "Test TIFF Preview Chat"})
    assert create_res.status_code == 200
    chat_id = create_res.json()["chat_id"]

    # 2. Upload dummy geotiff
    tif_bytes = _make_dummy_geotiff()
    res = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        data={"query": "Analyze water in this GeoTIFF scene"},
        files=[("image_a", ("sample_scene.tif", tif_bytes, "image/tiff"))],
    )
    assert res.status_code == 200

    # 3. Verify chat detail returns preview_url for TIFF attachment
    detail_res = client.get(f"/api/v1/chats/{chat_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert len(detail["messages"]) >= 2
    user_msg = detail["messages"][0]
    assert len(user_msg["attachments"]) == 1
    att = user_msg["attachments"][0]
    assert att["name"] == "sample_scene.tif"
    assert att["url"].endswith(".tif")
    assert att.get("preview_url") is not None
    assert att["preview_url"].endswith("_preview.png")

    # 4. Verify preview endpoint can serve this image preview
    preview_res = client.get(f"/api/v1/preview?path={att['url']}")
    assert preview_res.status_code == 200
    assert preview_res.headers["content-type"] == "image/png"
    # Ensure it's valid PNG data
    img = Image.open(io.BytesIO(preview_res.content))
    assert img.format == "PNG"
    assert img.size[0] > 0
    assert img.size[1] > 0
