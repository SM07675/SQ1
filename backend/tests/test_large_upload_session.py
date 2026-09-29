from unittest.mock import Mock

from fastapi.testclient import TestClient

from app import main


def test_large_tiff_upload_session_preserves_original_size_and_origin(monkeypatch):
    blob = Mock()
    blob.create_resumable_upload_session.return_value = "https://storage.googleapis.com/upload/session"
    monkeypatch.setattr(main, "_upload_blob", lambda upload_id: blob)
    client = TestClient(main.app)

    response = client.post(
        "/api/v1/uploads/session",
        json={"filename": "scene.tif", "size": 35 * 1024 * 1024},
        headers={"Origin": "https://frontend-chi-mauve-51.vercel.app"},
    )

    assert response.status_code == 200
    assert response.json()["upload_id"].endswith(".tif")
    blob.create_resumable_upload_session.assert_called_once_with(
        content_type="image/tiff",
        size=35 * 1024 * 1024,
        origin="https://frontend-chi-mauve-51.vercel.app",
        if_generation_match=0,
    )


def test_large_upload_rejects_unsupported_format_and_excess_size():
    client = TestClient(main.app)
    assert client.post("/api/v1/uploads/session", json={"filename": "scene.exe", "size": 100}).status_code == 415
    assert client.post("/api/v1/uploads/session", json={"filename": "scene.tif", "size": 257 * 1024 * 1024}).status_code == 413
