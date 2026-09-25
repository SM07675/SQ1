from __future__ import annotations

import gc
import time
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Any, Callable, Iterator


class ModelRuntimeManager:
    """Lazy, bounded runtime for a single accelerator or CPU-only deployment."""

    def __init__(self, max_resident: int = 2, device: str = "auto") -> None:
        self.max_resident = max(1, int(max_resident))
        self.device = self._select_device(device)
        self._models: OrderedDict[str, Any] = OrderedDict()
        self._health: dict[str, dict[str, Any]] = {}
        self._lock = RLock()

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {requested}")
        if requested == "cpu":
            return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    @contextmanager
    def acquire(self, model_id: str, loader: Callable[[], Any]) -> Iterator[Any]:
        started = time.perf_counter()
        with self._lock:
            try:
                if model_id not in self._models:
                    while len(self._models) >= self.max_resident:
                        oldest, _ = self._models.popitem(last=False)
                        self._health.setdefault(oldest, {})["resident"] = False
                        self._cleanup_device()
                    self._models[model_id] = loader()
                self._models.move_to_end(model_id)
                self._health[model_id] = {
                    "status": "READY",
                    "resident": True,
                    "device": self.device,
                    "load_ms": round((time.perf_counter() - started) * 1000, 2),
                }
                yield self._models[model_id]
            except Exception as exc:
                self._models.pop(model_id, None)
                self._health[model_id] = {"status": "LOAD_FAILED", "resident": False, "reason": str(exc)}
                self._cleanup_device()
                raise

    def unload(self, model_id: str) -> None:
        with self._lock:
            self._models.pop(model_id, None)
            self._health.setdefault(model_id, {})["resident"] = False
            self._cleanup_device()

    @staticmethod
    def _cleanup_device() -> None:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def snapshot(self) -> dict[str, Any]:
        allocated = 0
        try:
            import torch
            if torch.cuda.is_available():
                allocated = int(torch.cuda.memory_allocated())
        except ImportError:
            pass
        return {
            "device": self.device,
            "max_resident": self.max_resident,
            "resident_models": list(self._models),
            "allocated_vram_bytes": allocated,
            "models": dict(self._health),
        }
