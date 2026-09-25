from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class VLMConnector:
    """Provider-neutral multipart connector for LLaVA, BLIP-2 or managed MLLM services."""

    endpoint: str
    api_key: str | None = None
    model_name: str = "external-vlm"

    async def infer(
        self,
        *,
        query: str,
        image_paths: list[Path],
        tile_paths: list[Path],
        metadata: list[dict[str, Any]],
        timeout_seconds: float = 180,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "model": self.model_name,
            "metadata": metadata,
            "image_count": len(image_paths),
            "tile_count": len(tile_paths),
            "image_paths": [str(p) for p in image_paths],
            "tile_paths": [str(p) for p in tile_paths],
            "response_schema": {
                "answer": "string",
                "confidence": "number 0..1",
                "token_logprobs": "optional number[]",
                "supports_claim": "optional boolean",
                "boxes": "optional [x1,y1,x2,y2][] normalized 0..1",
                "metrics": "optional object",
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        upload_paths = tile_paths[:16] or image_paths[:2]
        files = [
            ("images", (path.name, path.read_bytes(), "image/png" if path.suffix.lower() == ".png" else "image/tiff"))
            for path in upload_paths
        ]
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                self.endpoint.rstrip("/") + "/infer",
                data={"request": json.dumps(payload)},
                files=files,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
        if not isinstance(result.get("answer"), str):
            raise ValueError("VLM response must contain a string 'answer'")
        confidence = float(result.get("confidence", 0.5))
        if not 0 <= confidence <= 1:
            raise ValueError("VLM confidence must be between 0 and 1")
        result.update({
            "available": True,
            "model": self.model_name,
            "confidence": confidence,
            "duration_ms": round((time.perf_counter() - started) * 1000),
        })
        return result
