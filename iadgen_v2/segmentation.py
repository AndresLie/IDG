from __future__ import annotations

import math

import numpy as np


def segmentation_metrics(prediction: np.ndarray, truth: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    score_map = np.asarray(prediction, dtype=np.float32)
    truth_map = (np.asarray(truth) > 0).astype(np.uint8)
    scores = score_map.reshape(-1)
    labels = truth_map.reshape(-1)
    binary = scores >= threshold
    intersection = int(np.logical_and(binary, labels).sum())
    predicted = int(binary.sum())
    positive = int(labels.sum())
    union = int(np.logical_or(binary, labels).sum())
    return {
        "pixel_auroc": _auroc(scores, labels),
        "aupro": _aupro(score_map, truth_map),
        "iou": intersection / union if union else math.nan,
        "dice": (2 * intersection) / (predicted + positive) if predicted + positive else math.nan,
    }


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    positive_rank_sum = float(ranks[labels[order] == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _aupro(scores: np.ndarray, truth: np.ndarray, max_fpr: float = 0.3) -> float:
    if scores.ndim != 2 or truth.ndim != 2:
        return math.nan
    components = _components(truth)
    negative = truth == 0
    if not components or not negative.any():
        return math.nan
    points: list[tuple[float, float]] = []
    for threshold in np.linspace(1.0, 0.0, 101):
        predicted = scores >= threshold
        fpr = float(predicted[negative].mean())
        pro = float(np.mean([predicted[component].mean() for component in components]))
        if fpr <= max_fpr:
            points.append((fpr, pro))
    if not points:
        return 0.0
    points.extend([(0.0, 0.0), (max_fpr, points[-1][1])])
    ordered = sorted(points)
    fpr = np.asarray([point[0] for point in ordered], dtype=np.float32)
    pro = np.asarray([point[1] for point in ordered], dtype=np.float32)
    return _integrate(pro, fpr) / max_fpr


def _components(truth: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(truth, dtype=bool)
    result: list[np.ndarray] = []
    height, width = truth.shape
    for y in range(height):
        for x in range(width):
            if truth[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and truth[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            component = np.zeros_like(truth, dtype=bool)
            ys, xs = zip(*pixels)
            component[np.asarray(ys), np.asarray(xs)] = True
            result.append(component)
    return result


def _integrate(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5))
