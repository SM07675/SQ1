import pytest
from unittest.mock import AsyncMock, patch
from satquery_engine.services.model_registry import ModelRegistry


def test_model_registry_capabilities():
    registry = ModelRegistry()
    caps = {item["name"]: item for item in registry.capabilities()}
    
    assert "earthdial" in caps
    assert "croma" in caps
    assert "change" in caps
    assert "remoteclip" in caps
    assert "vlm" in caps


@pytest.mark.asyncio
async def test_invoke_when_unconfigured():
    with patch("satquery_engine.services.model_registry.settings") as mock_settings:
        mock_settings.earthdial_endpoint = None
        mock_settings.croma_endpoint = None
        mock_settings.change_endpoint = None
        mock_settings.remoteclip_endpoint = None
        
        reg = ModelRegistry()
        result = await reg.invoke("change", {"before_path": "a.tif", "after_path": "b.tif"})
        assert result["available"] is False
        assert "SATQUERY_CHANGE_ENDPOINT is not configured" in result["reason"]
