from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def seed_everything(seed: int = 20260922) -> None:
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_geographic_split(records: Iterable[dict]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        geography, split = str(record["geography"]), str(record["split"])
        previous = seen.setdefault(geography, split)
        if previous != split:
            raise ValueError(f"Geography '{geography}' leaks across {previous} and {split} splits.")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_loss(logits, target, class_weights, ignore_index: int = 255, focal_gamma: float | None = None):
    import torch
    import torch.nn.functional as functional
    ce = functional.cross_entropy(logits, target, weight=class_weights, ignore_index=ignore_index, reduction="none")
    if focal_gamma is not None:
        ce = ((1 - torch.exp(-ce)) ** focal_gamma) * ce
    valid = target != ignore_index
    ce = ce[valid].mean() if valid.any() else logits.sum() * 0
    probabilities = torch.softmax(logits, dim=1)
    safe_target = target.masked_fill(~valid, 0)
    one_hot = functional.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).float() * valid[:, None]
    intersection = (probabilities * one_hot).sum((0, 2, 3))
    denominator = (probabilities * valid[:, None]).sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    dice = 1 - ((2 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return 0.5 * ce + 0.5 * dice


def save_checkpoint(path: str | Path, model, optimizer, scheduler, scaler, epoch: int, best_metric: float, stale: int, config: dict) -> None:
    import torch
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None, "scaler": scaler.state_dict() if scaler else None, "epoch": epoch, "best_metric": best_metric, "stale": stale, "config": config, "rng": {"torch": torch.random.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}}
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None) -> dict:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    for object_, key in ((optimizer, "optimizer"), (scheduler, "scheduler"), (scaler, "scaler")):
        if object_ is not None and payload.get(key) is not None:
            object_.load_state_dict(payload[key])
    return payload


def tune_minimum_confidence(probabilities: np.ndarray, targets: np.ndarray, candidates: Iterable[float], ignore_index: int = 255) -> dict[str, float]:
    from .evaluation import confusion_matrix, metrics_from_confusion
    best = {"threshold": 0.0, "macro_iou": -1.0, "coverage": 0.0, "objective": -1.0}
    prediction = probabilities.argmax(1)
    confidence = probabilities.max(1)
    for threshold in candidates:
        accepted = (confidence >= threshold) & (targets != ignore_index)
        adjusted_target = np.where(accepted, targets, ignore_index)
        score = metrics_from_confusion(confusion_matrix(adjusted_target, prediction, probabilities.shape[1], ignore_index))["macro_iou"]
        coverage = float(accepted.sum() / max(1, np.count_nonzero(targets != ignore_index)))
        objective = float(score * coverage)
        if objective > best["objective"]:
            best = {"threshold": float(threshold), "macro_iou": float(score), "coverage": coverage, "objective": objective}
    return best


def export_training_bundle(destination: str | Path, files: Iterable[str | Path], manifest: dict) -> Path:
    import zipfile
    destination = Path(destination)
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            file = Path(file)
            archive.write(file, arcname=file.name)
        archive.write(manifest_path, arcname="export_manifest.json")
    return destination
