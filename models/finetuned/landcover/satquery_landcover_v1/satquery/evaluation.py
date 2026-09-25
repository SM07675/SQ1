from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def confusion_matrix(target: np.ndarray, prediction: np.ndarray, class_count: int, ignore_index: int = 255) -> np.ndarray:
    valid = (target != ignore_index) & (target >= 0) & (target < class_count)
    encoded = target[valid].astype("int64") * class_count + prediction[valid].astype("int64")
    return np.bincount(encoded, minlength=class_count * class_count).reshape(class_count, class_count)


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    matrix = matrix.astype("float64")
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    tn = matrix.sum() - tp - fp - fn
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    iou = divide(tp, tp + fp + fn)
    f1 = divide(2 * precision * recall, precision + recall)
    specificity = divide(tn, tn + fp)
    return {
        "per_class_iou": iou.tolist(), "per_class_precision": precision.tolist(), "per_class_recall": recall.tolist(), "per_class_f1": f1.tolist(),
        "macro_iou": float(iou.mean()), "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()), "specificity": specificity.tolist(),
        "confusion_matrix": matrix.astype("int64").tolist(),
    }


def water_metrics(target: np.ndarray, prediction: np.ndarray, dark_land: np.ndarray | None = None, valid: np.ndarray | None = None) -> dict[str, float]:
    mask = np.ones(target.shape, bool) if valid is None else valid.astype(bool)
    truth, pred = target.astype(bool) & mask, prediction.astype(bool) & mask
    tp = int(np.count_nonzero(truth & pred)); fp = int(np.count_nonzero(~truth & pred & mask))
    fn = int(np.count_nonzero(truth & ~pred & mask)); tn = int(np.count_nonzero(~truth & ~pred & mask))
    ratio = lambda a, b: float(a / b) if b else 0.0
    result = {"iou": ratio(tp, tp + fp + fn), "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn), "specificity": ratio(tn, tn + fp)}
    if dark_land is not None:
        negative = dark_land.astype(bool) & ~truth & mask
        result["dark_land_shadow_false_positive_rate"] = ratio(np.count_nonzero(pred & negative), np.count_nonzero(negative))
    return result


def _instance_iou_matrix(target: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_ids = np.unique(target); target_ids = target_ids[target_ids > 0]
    pred_ids = np.unique(prediction); pred_ids = pred_ids[pred_ids > 0]
    matrix = np.zeros((len(target_ids), len(pred_ids)), dtype="float64")
    for row, target_id in enumerate(target_ids):
        truth = target == target_id
        overlaps = np.unique(prediction[truth]); overlaps = overlaps[overlaps > 0]
        for pred_id in overlaps:
            column = int(np.searchsorted(pred_ids, pred_id))
            candidate = prediction == pred_id
            matrix[row, column] = np.count_nonzero(truth & candidate) / np.count_nonzero(truth | candidate)
    return matrix, target_ids, pred_ids


def building_metrics(target: np.ndarray, prediction: np.ndarray, semantic_target: np.ndarray | None = None, semantic_probability: np.ndarray | None = None, threshold: float = 0.5) -> dict[str, float]:
    ious, target_ids, pred_ids = _instance_iou_matrix(target, prediction)
    matches: list[float] = []
    if ious.size:
        rows, columns = linear_sum_assignment(-ious)
        matches = [float(ious[r, c]) for r, c in zip(rows, columns) if ious[r, c] >= threshold]
    tp, fp, fn = len(matches), len(pred_ids) - len(matches), len(target_ids) - len(matches)
    ratio = lambda a, b: float(a / b) if b else 0.0
    truth_semantic = target > 0 if semantic_target is None else semantic_target.astype(bool)
    pred_semantic = prediction > 0 if semantic_probability is None else semantic_probability >= 0.5
    intersection = np.count_nonzero(truth_semantic & pred_semantic)
    union = np.count_nonzero(truth_semantic | pred_semantic)
    count_error = abs(len(pred_ids) - len(target_ids))
    return {
        "semantic_iou": ratio(intersection, union), "instance_precision@0.5": ratio(tp, tp + fp), "instance_recall@0.5": ratio(tp, tp + fn),
        "instance_f1@0.5": ratio(2 * tp, 2 * tp + fp + fn), "matched_instance_iou": float(np.mean(matches)) if matches else 0.0,
        "count_mae": float(count_error), "relative_count_error": ratio(count_error, len(target_ids)), "target_count": int(len(target_ids)), "predicted_count": int(len(pred_ids)),
    }


def write_scene_report(path: str | Path, scene_metrics: dict[str, dict[str, Any]], global_metrics: dict[str, Any]) -> None:
    payload = {"global": global_metrics, "by_scene_or_geography": scene_metrics, "aggregation_note": "Global values do not replace scene/geography-stratified reporting."}
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def rank_worst_failures(scene_metrics: dict[str, dict[str, float]], metric: str, count: int = 10) -> list[str]:
    return [key for key, _ in sorted(scene_metrics.items(), key=lambda item: item[1].get(metric, float("inf")))[:count]]


def save_failure_visualizations(output_dir: str | Path, cases: list[dict[str, Any]], count_per_category: int = 3) -> list[str]:
    """Save RGB/target/prediction panels for required failure categories."""
    from PIL import Image, ImageDraw

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    palette = np.array([[30,30,30],[220,70,70],[60,170,70],[220,190,60],[190,140,80],[50,120,220],[150,150,150],[255,0,255]], dtype="uint8")
    paths: list[str] = []
    categories = ("shadow", "cloud", "dense_buildings", "tiny_buildings", "mixed_shoreline", "tile_boundary")
    for category in categories:
        selected = sorted((case for case in cases if case.get("category") == category), key=lambda case: float(case["score"]))[:count_per_category]
        for index, case in enumerate(selected):
            rgb = np.asarray(case["rgb"])
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0, 1) * 255).astype("uint8")
            target = palette[np.clip(np.asarray(case["target"], dtype="int64"), 0, len(palette)-1)]
            prediction = palette[np.clip(np.asarray(case["prediction"], dtype="int64"), 0, len(palette)-1)]
            panel = Image.new("RGB", (rgb.shape[1] * 3, rgb.shape[0] + 24), "white")
            for column, array in enumerate((rgb, target, prediction)):
                panel.paste(Image.fromarray(array, "RGB"), (column * rgb.shape[1], 24))
            ImageDraw.Draw(panel).text((4, 4), f"{category} | RGB / target / prediction | score={float(case['score']):.4f}", fill="black")
            path = destination / f"{category}_{index:02d}.png"
            panel.save(path)
            paths.append(str(path))
    return paths
