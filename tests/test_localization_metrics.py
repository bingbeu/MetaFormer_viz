import numpy as np

from localization_metrics import (
    aggregate_evidence_maps,
    box_iou_from_masks,
    evaluate_evidence_consensus,
    evaluate_heatmap,
    evaluate_part_points,
    pointing_game,
    select_top_evidence_tokens,
    square_deletion_mask,
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


def test_evidence_consensus_allows_shared_peaks():
    maps = np.zeros((4, 2, 2), dtype=float)
    maps[:3, 0, 1] = 1.0
    maps[3, 1, 0] = 1.0
    foreground = np.zeros((8, 8), dtype=bool)
    foreground[:4, 4:] = True
    result = evaluate_evidence_consensus(maps, foreground)
    assert result["consensus_ratio"] == 0.75
    assert result["unique_peak_ratio"] == 0.5
    assert result["consensus_foreground_hit"] == 1.0
    assert result["consensus_x"] == 6.0
    assert result["consensus_y"] == 2.0


def test_square_deletion_mask_keeps_constant_area_at_border():
    mask = square_deletion_mask((10, 10), (0.0, 0.0), side_fraction=0.4)
    assert mask.shape == (10, 10)
    assert int(mask.sum()) == 16
    assert mask[0, 0]


def test_evidence_aggregation_uses_all_tokens():
    maps = np.asarray([
        [[0.0, 1.0], [0.0, 0.0]],
        [[0.0, 0.0], [2.0, 0.0]],
    ])
    assert np.allclose(aggregate_evidence_maps(maps, "mean"), maps.mean(axis=0))
    assert np.allclose(aggregate_evidence_maps(maps, "max"), maps.max(axis=0))


def test_top_evidence_selection_is_peak_ranked_and_deterministic():
    maps = np.zeros((4, 2, 2), dtype=float)
    maps[0, 0, 0] = 0.5
    maps[1, 0, 1] = 0.8
    maps[2, 1, 0] = 0.8
    maps[3, 1, 1] = 0.2
    assert select_top_evidence_tokens(maps, 3).tolist() == [1, 2, 0]
    assert select_top_evidence_tokens(maps, 99).tolist() == [1, 2, 0, 3]
