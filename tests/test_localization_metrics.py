import numpy as np

from localization_metrics import (
    box_iou_from_masks,
    evaluate_heatmap,
    evaluate_part_points,
    pointing_game,
    top_fraction_mask,
)


def test_pointing_and_foreground_energy():
    heatmap = np.zeros((4, 4), dtype=float)
    heatmap[1, 1] = 10.0
    foreground = np.zeros((4, 4), dtype=bool)
    foreground[:2, :2] = True
    metrics = evaluate_heatmap(heatmap, foreground, top_fraction=0.25)
    assert metrics["pointing_game"] == 1.0
    assert metrics["foreground_energy"] == 1.0
    assert metrics["foreground_concentration_gain"] == 4.0


def test_exact_top_fraction_count_with_ties():
    mask = top_fraction_mask(np.ones((10, 10)), 0.20)
    assert int(mask.sum()) == 20


def test_pointing_game_uses_one_deterministic_maximum():
    heatmap = np.ones((3, 3))
    foreground = np.zeros((3, 3), dtype=bool)
    foreground[2, 2] = True
    assert pointing_game(heatmap, foreground) == 0.0


def test_box_iou():
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[1:5, 1:5] = True
    b[3:7, 3:7] = True
    assert np.isclose(box_iou_from_masks(a, b), 4.0 / 28.0)


def test_part_matching_perfect():
    points = np.asarray([[5.0, 5.0], [15.0, 15.0]])
    result = evaluate_part_points(points, points[::-1], normalization_length=20.0)
    assert result["matched_mean_nme"] == 0.0
    assert result["matched_pck_0p1"] == 1.0
    assert result["gt_coverage_pck_0p1"] == 1.0
