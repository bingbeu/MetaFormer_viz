#!/usr/bin/env python3
"""CPU-only smoke test for publication plotting and colour policy."""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_visualization.visualize_part_evidence import (  # noqa: E402
    ALLOWED_CMAPS,
    CMAP_AGREEMENT,
    CMAP_ATTENTION,
    CMAP_DIFFERENCE,
    CMAP_MATRIX,
    configure_publication_style,
    plot_overview,
    plot_part_evidence,
    plot_semantic_grounding,
    plot_teacher_student,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/curv_part_visual_self_check")
    return parser.parse_args()


def gaussian(grid_y, grid_x, center_y, center_x, sigma):
    return np.exp(
        -((grid_y - center_y) ** 2 + (grid_x - center_x) ** 2) / (2.0 * sigma**2)
    )


def main():
    args = parse_args()
    configure_publication_style()
    expected = {CMAP_ATTENTION, CMAP_AGREEMENT, CMAP_DIFFERENCE, CMAP_MATRIX}
    if expected != ALLOWED_CMAPS:
        raise AssertionError("Colormap policy is internally inconsistent")
    for cmap in ALLOWED_CMAPS:
        if cmap not in matplotlib.colormaps:
            raise AssertionError("Unavailable colormap: {}".format(cmap))
        if cmap.lower() in {"jet", "rainbow", "gist_rainbow", "nipy_spectral"}:
            raise AssertionError("Non-publication colormap configured: {}".format(cmap))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height = width = 224
    yy_image, xx_image = np.mgrid[0:height, 0:width]
    image = np.stack(
        [
            0.25 + 0.55 * xx_image / width,
            0.30 + 0.45 * yy_image / height,
            np.full((height, width), 0.55),
        ],
        axis=-1,
    )
    image = np.clip(image, 0.0, 1.0)

    side = 24
    yy, xx = np.mgrid[0:side, 0:side]
    centers = [(8, 8), (8, 8), (8, 9), (15, 15), (15, 15), (12, 7), (7, 16), (16, 8)]
    parts = []
    for index, (cy, cx) in enumerate(centers):
        value = gaussian(yy, xx, cy, cx, 2.2 + 0.1 * index) + 1e-4
        value /= value.sum()
        parts.append(value)
    parts = np.asarray(parts)
    student = gaussian(yy, xx, 9, 9, 3.0) + 0.4 * gaussian(yy, xx, 15, 15, 2.5)
    teacher = gaussian(yy, xx, 9, 8.5, 2.8) + 0.35 * gaussian(yy, xx, 15, 15, 2.6)

    metrics, _, agreement, coverage, _ = plot_part_evidence(
        image, parts, output_dir / "smoke", ("png",), 0.55, 0.10
    )
    fidelity = plot_teacher_student(
        image, student, teacher, output_dir / "smoke", ("png",), 0.55
    )
    semantic_logits = np.linspace(-1.0, 1.0, 8 * 32).reshape(8, 32)
    semantic_attention = np.exp(semantic_logits)
    semantic_attention /= semantic_attention.sum(axis=1, keepdims=True)
    plot_semantic_grounding(
        semantic_attention,
        "synthetic caption for plotting validation",
        output_dir / "smoke",
        ("png",),
    )
    plot_overview(
        image,
        student,
        teacher,
        agreement,
        coverage,
        output_dir / "smoke",
        ("png",),
        0.55,
    )

    expected_files = [
        "smoke_part_attention.png",
        "smoke_part_overlap.png",
        "smoke_teacher_student.png",
        "smoke_semantic_grounding.png",
        "smoke_overview.png",
    ]
    for filename in expected_files:
        path = output_dir / filename
        if not path.exists() or path.stat().st_size < 1000:
            raise AssertionError("Missing or empty self-check output: {}".format(path))
    if not 0.0 <= metrics["mean_pairwise_overlap"] <= 1.0:
        raise AssertionError("Part-overlap metric outside [0, 1]")
    if not -1.0 <= fidelity["teacher_student_spearman"] <= 1.0:
        raise AssertionError("Spearman correlation outside [-1, 1]")
    print("[PASS] plotting, metrics, output files, and colour policy")
    print("[PASS] publication colormaps: {}".format(", ".join(sorted(ALLOWED_CMAPS))))
    print("[PASS] preview directory: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
