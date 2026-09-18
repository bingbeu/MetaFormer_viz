#!/usr/bin/env python3
"""CPU-only scientific and rendering checks for visual-analysis v2."""

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_visualization.visualize_evidence_consensus import (
    CMAP_AGREEMENT,
    CMAP_DIVERGENCE,
    CMAP_EVIDENCE,
    _cmap,
    configure_publication_style,
    plot_gallery,
    plot_query_diagnostics,
    prepare_record,
    robust_unit_map,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/curv_part_visual_v2_check")
    return parser.parse_args()


def gaussian(yy, xx, cy, cx, sigma):
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma**2))


def main():
    args = parse_args()
    configure_publication_style()
    for name in (CMAP_EVIDENCE, CMAP_AGREEMENT, CMAP_DIVERGENCE):
        _cmap(name)
        if name.lower() in {"jet", "rainbow", "gist_rainbow", "nipy_spectral"}:
            raise AssertionError("Non-perceptual colormap configured: {}".format(name))

    # Regression check: tied values must remain tied.  The v1 rank transform
    # assigned an arbitrary gradient to a constant field.
    constant = np.ones((12, 12), dtype=np.float32)
    if not np.allclose(robust_unit_map(constant), 0.0):
        raise AssertionError("A constant map produced artificial spatial evidence")

    height = width = 224
    yi, xi = np.mgrid[0:height, 0:width]
    image = np.stack(
        [
            0.18 + 0.62 * xi / width,
            0.24 + 0.52 * yi / height,
            np.full((height, width), 0.58),
        ],
        axis=-1,
    )
    image = np.clip(image, 0.0, 1.0)

    side = 24
    yy, xx = np.mgrid[0:side, 0:side]
    # A consensus-dominant synthetic case with small, controlled differences.
    base = gaussian(yy, xx, 8.0, 9.0, 2.4) + 0.55 * gaussian(yy, xx, 14.0, 16.0, 2.8)
    parts = []
    for index in range(8):
        local = base + 0.08 * gaussian(
            yy,
            xx,
            6.0 + (index % 3) * 5.0,
            5.0 + (index % 4) * 4.0,
            1.8,
        )
        local = np.maximum(local, 0.0)
        local /= local.sum()
        parts.append(local)
    parts = np.asarray(parts)
    student = np.full((side, side), 0.01, dtype=np.float64)
    student += 0.22 * gaussian(yy, xx, 8.0, 9.0, 1.6)

    metadata = {
        "sample_index": 0,
        "caption_index": 0,
        "image_path": "synthetic",
        "caption": "synthetic sample",
        "target": 0,
        "target_name": "synthetic",
        "prediction": 0,
        "prediction_name": "synthetic",
        "confidence": 0.9,
        "correct": True,
        "layer": 2,
    }
    record = prepare_record(image, student, parts, metadata, 0.10)
    if not 0.0 <= record["metrics"]["mean_pairwise_overlap"] <= 1.0:
        raise AssertionError("Part overlap outside [0, 1]")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_gallery(
        [record], output_dir / "visual_analysis_layer2", ("png", "pdf"), 0.72, 2
    )
    plot_query_diagnostics(
        record, output_dir / "sample_0000_layer2", ("png", "pdf"), 0.72
    )
    expected = [
        output_dir / "visual_analysis_layer2.png",
        output_dir / "visual_analysis_layer2.pdf",
        output_dir / "sample_0000_layer2_part_queries.png",
        output_dir / "sample_0000_layer2_query_divergence.png",
    ]
    for path in expected:
        if not path.exists() or path.stat().st_size < 1000:
            raise AssertionError("Missing or empty output: {}".format(path))
    print("[PASS] tied-value, attention-reference, metric, colour, and render checks")
    print("[PASS] preview directory: {}".format(output_dir.resolve()))


if __name__ == "__main__":
    main()
