"""One runtime device policy for the active PyTorch and ONNX model routes."""
from __future__ import annotations

import os


def torch_device() -> str:
    import torch

    requested = os.getenv("SATQUERY_DEVICE", "auto").lower()
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("SATQUERY_DEVICE must be auto, cuda, or cpu")
    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but PyTorch cannot access the GPU")
    return "cpu"


def onnx_providers() -> list[str]:
    import onnxruntime as ort

    if torch_device() == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA is available, but ONNX Runtime has no CUDA execution provider")
        try:
            ort.preload_dlls()
        except AttributeError:
            pass
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
