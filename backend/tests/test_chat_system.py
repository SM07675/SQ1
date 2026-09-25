import io
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_chat_lifecycle():
    # 1. Create chat
    create_res = client.post("/api/v1/chats", json={"title": "Test Water Chat"})
    assert create_res.status_code == 200
    chat_data = create_res.json()
    chat_id = chat_data["chat_id"]
    assert chat_data["title"] == "Test Water Chat"

    # 2. List chats
    list_res = client.get("/api/v1/chats")
    assert list_res.status_code == 200
    chats = list_res.json()
    assert any(c["chat_id"] == chat_id for c in chats)

    # 3. Rename chat
    rename_res = client.patch(f"/api/v1/chats/{chat_id}", json={"title": "Renamed Chat"})
    assert rename_res.status_code == 200
    assert rename_res.json()["title"] == "Renamed Chat"

    # 4. Upload image and post initial query
    img = Image.new("RGB", (100, 100), color=(30, 80, 140))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    post_msg_res = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        data={"query": "Detect water in this coastal area"},
        files=[("files", ("test.png", buf.getvalue(), "image/png"))],
    )
    assert post_msg_res.status_code == 200
    msg_data = post_msg_res.json()
    assert msg_data["chat_id"] == chat_id
    assert msg_data["role"] == "assistant"
    assert msg_data["result"] is not None

    # Check that chat title was automatically updated to something meaningful
    detail_res = client.get(f"/api/v1/chats/{chat_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert len(detail["messages"]) == 2
    assert len(detail["images"]) == 1

    # 5. Follow-up query WITHOUT uploading image again
    follow_up_res = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        data={"query": "Where is the largest water body located and how big is it?"},
    )
    assert follow_up_res.status_code == 200
    fu_data = follow_up_res.json()
    assert fu_data["role"] == "assistant"
    fu_text = fu_data["content"]
    assert len(fu_text) > 0
    assert "water" in fu_text.lower() or "area" in fu_text.lower() or "region" in fu_text.lower()

    # 5b. Follow-up query for evidence/why
    why_res = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        data={"query": "Why did you detect this as water?"},
    )
    assert why_res.status_code == 200
    why_text = why_res.json()["content"]
    assert "optical" in why_text.lower() or "spectral" in why_text.lower() or "water" in why_text.lower()

    # 5c. Follow-up query for confidence
    conf_res = client.post(
        f"/api/v1/chats/{chat_id}/messages",
        data={"query": "What is the confidence score?"},
    )
    assert conf_res.status_code == 200
    assert "%" in conf_res.json()["content"] or "confidence" in conf_res.json()["content"].lower()

    # Detail now has 8 messages (4 user + 4 assistant)
    detail2 = client.get(f"/api/v1/chats/{chat_id}").json()
    assert len(detail2["messages"]) == 8

    # 6. Delete chat
    del_res = client.delete(f"/api/v1/chats/{chat_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True

    # Verify not found
    get_after_del = client.get(f"/api/v1/chats/{chat_id}")
    assert get_after_del.status_code == 404
