"""Quantitative localization metrics for Curv-Part heatmaps.

The functions in this module are model-agnostic and operate on NumPy arrays.
They intentionally distinguish pixel-mask IoU from bounding-box IoU so that
results obtained with CUB's box annotations are not reported as segmentation
IoU.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


EPS = 1e-12


def _finite_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).squeeze()
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D map, got shape={x.shape}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def nonnegative_heatmap(x: np.ndarray) -> np.ndarray:
    """Return a finite, non-negative heatmap without changing its ordering."""
    x = _finite_2d(x)
    lo = float(x.min())
    if lo < 0.0:
        x = x - lo
    return np.maximum(x, 0.0)


def top_fraction_mask(heatmap: np.ndarray, fraction: float) -> np.ndarray:
    """Select exactly the highest-scoring fraction of pixels.

    Selecting an exact count avoids quantile/tie behaviour that can otherwise
    turn a constant heatmap into a full-image prediction.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    h = nonnegative_heatmap(heatmap)
    flat = h.reshape(-1)
    k = max(1, int(round(flat.size * fraction)))
    if k >= flat.size:
        return np.ones_like(h, dtype=bool)
    ids = np.argpartition(flat, flat.size - k)[-k:]
    out = np.zeros(flat.size, dtype=bool)
    out[ids] = True
    return out.reshape(h.shape)


def pointing_game(heatmap: np.ndarray, foreground_mask: np.ndarray) -> float:
    """Return 1 when the single global-maximum pixel lies in foreground."""
    h = nonnegative_heatmap(heatmap)
    fg = np.asarray(foreground_mask, dtype=bool)
    if fg.shape != h.shape or not fg.any():
        return float("nan")
    row, col = np.unravel_index(int(np.argmax(h)), h.shape)
    return float(fg[row, col])


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction, dtype=bool)
    tgt = np.asarray(target, dtype=bool)
    if pred.shape != tgt.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, target={tgt.shape}")
    union = np.logical_or(pred, tgt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, tgt).sum() / union)


def mask_to_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return an inclusive-exclusive (x1, y1, x2, y2) box."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_iou_from_masks(prediction: np.ndarray, target: np.ndarray) -> float:
    a, b = mask_to_box(prediction), mask_to_box(target)
    if a is None or b is None:
        return float("nan")
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / max(area_a + area_b - inter, EPS))


def foreground_energy_fraction(heatmap: np.ndarray, foreground_mask: np.ndarray) -> float:
    h = nonnegative_heatmap(heatmap)
    fg = np.asarray(foreground_mask, dtype=bool)
    if fg.shape != h.shape or not fg.any():
        return float("nan")
    total = float(h.sum())
    if total <= EPS:
        # A constant zero map carries no localization information; treating it
        # as uniform makes the concentration gain correctly equal to one.
        return float(fg.mean())
    return float(h[fg].sum() / total)


def foreground_concentration_gain(heatmap: np.ndarray, foreground_mask: np.ndarray) -> float:
    fg = np.asarray(foreground_mask, dtype=bool)
    area_fraction = float(fg.mean())
    if area_fraction <= EPS:
        return float("nan")
    return foreground_energy_fraction(heatmap, fg) / area_fraction


def evaluate_heatmap(
    heatmap: np.ndarray,
    foreground_mask: np.ndarray,
    top_fraction: float = 0.20,
) -> Dict[str, float]:
    """Evaluate one dense heatmap against a foreground mask or box mask."""
    fg = np.asarray(foreground_mask, dtype=bool)
    pred = top_fraction_mask(heatmap, top_fraction)
    box_iou = box_iou_from_masks(pred, fg)
    return {
        "pointing_game": pointing_game(heatmap, fg),
        "pixel_iou": binary_iou(pred, fg),
        "pred_box_iou": box_iou,
        "loc_acc_iou50": float(box_iou >= 0.5) if np.isfinite(box_iou) else float("nan"),
        "foreground_energy": foreground_energy_fraction(heatmap, fg),
        "foreground_concentration_gain": foreground_concentration_gain(heatmap, fg),
        "topk_foreground_precision": float(fg[pred].mean()) if pred.any() else float("nan"),
        "foreground_area_fraction": float(fg.mean()),
    }


def part_peak_points(part_maps: np.ndarray, image_shape: Sequence[int]) -> np.ndarray:
    """Convert P token heatmaps into P peak coordinates in image pixels."""
    maps = np.asarray(part_maps, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError(f"part_maps must be [P,H,W], got {maps.shape}")
    height, width = int(image_shape[0]), int(image_shape[1])
    _, gh, gw = maps.shape
    points = []
    for m in maps:
        token_id = int(np.nanargmax(np.nan_to_num(m, nan=-np.inf)))
        row, col = np.unravel_index(token_id, (gh, gw))
        points.append(((col + 0.5) * width / gw, (row + 0.5) * height / gh))
    return np.asarray(points, dtype=np.float64)


def evaluate_part_points(
    predicted_points: np.ndarray,
    gt_points: np.ndarray,
    normalization_length: float,
    thresholds: Iterable[float] = (0.10, 0.20),
) -> Dict[str, float]:
    """Evaluate discovered part peaks against visible CUB keypoints.

    Hungarian matching measures one-to-one part localization. Coverage and
    precision use nearest neighbours and therefore expose missed GT parts and
    redundant predicted parts separately.
    """
    pred = np.asarray(predicted_points, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_points, dtype=np.float64).reshape(-1, 2)
    if len(pred) == 0 or len(gt) == 0 or normalization_length <= EPS:
        return {"matched_mean_nme": float("nan")}

    distances = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
    pred_ids, gt_ids = linear_sum_assignment(distances)
    matched = distances[pred_ids, gt_ids] / float(normalization_length)
    nearest_gt = distances.min(axis=0) / float(normalization_length)
    nearest_pred = distances.min(axis=1) / float(normalization_length)

    out = {
        "matched_mean_nme": float(matched.mean()),
        "matched_median_nme": float(np.median(matched)),
    }
    for threshold in thresholds:
        suffix = str(float(threshold)).replace(".", "p")
        out[f"matched_pck_{suffix}"] = float((matched <= threshold).mean())
        out[f"gt_coverage_pck_{suffix}"] = float((nearest_gt <= threshold).mean())
        out[f"pred_precision_pck_{suffix}"] = float((nearest_pred <= threshold).mean())

    if len(pred) > 1:
        pairwise = np.linalg.norm(pred[:, None, :] - pred[None, :, :], axis=-1)
        upper = pairwise[np.triu_indices(len(pred), k=1)] / float(normalization_length)
        out["part_pairwise_distance"] = float(upper.mean())
    else:
        out["part_pairwise_distance"] = float("nan")
    rounded = np.round(pred, decimals=4)
    out["part_unique_peak_ratio"] = float(len(np.unique(rounded, axis=0)) / len(pred))
    return out
