#!/usr/bin/env python3
"""Paired bootstrap comparison of two localization evaluation directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_METRICS = (
    "classification_top1",
    "classification_top5",
    "part_attention_mean_pointing_game",
    "part_attention_mean_foreground_energy",
    "part_attention_mean_foreground_concentration_gain",
    "part_attention_mean_pixel_iou",
    "part_attention_mean_loc_acc_iou50",
    "part_attention_mean_pred_box_iou",
)


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "image_id" not in rows[0]:
        raise ValueError(f"{path} is empty or has no image_id column")
    result = {row["image_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate image_id values in {path}")
    return result


def paired_bootstrap(
    baseline: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
    metrics: tuple[str, ...] | list[str],
    samples: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    ids = sorted(set(baseline) & set(candidate), key=lambda value: int(value))
    if not ids:
        raise ValueError("the two runs have no common image_id values")
    if set(baseline) != set(candidate):
        raise ValueError("paired comparison requires identical image_id sets")
    rng = np.random.default_rng(seed)
    results = []
    for metric in metrics:
        missing = [name for name, rows in (("baseline", baseline), ("candidate", candidate))
                   if metric not in rows[ids[0]]]
        if missing:
            raise KeyError(f"metric {metric!r} missing from {', '.join(missing)} run")
        base = np.asarray([float(baseline[i][metric]) for i in ids], dtype=np.float64)
        cand = np.asarray([float(candidate[i][metric]) for i in ids], dtype=np.float64)
        valid = np.isfinite(base) & np.isfinite(cand)
        delta = cand[valid] - base[valid]
        if not len(delta):
            continue
        # Chunk resamples so a full 5,794-image, 10k-bootstrap comparison does
        # not allocate a ~460 MB integer index matrix.
        boot = np.empty(samples, dtype=np.float64)
        chunk = 256
        for start in range(0, samples, chunk):
            stop = min(start + chunk, samples)
            draws = rng.integers(0, len(delta), size=(stop - start, len(delta)))
            boot[start:stop] = delta[draws].mean(axis=1)
        low, high = np.quantile(boot, (0.025, 0.975))
        results.append({
            "metric": metric,
            "n": int(len(delta)),
            "baseline_mean": float(base[valid].mean()),
            "candidate_mean": float(cand[valid].mean()),
            "delta": float(delta.mean()),
            "paired_ci95_low": float(low),
            "paired_ci95_high": float(high),
        })
    return results


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline output directory")
    parser.add_argument("--candidate", required=True, help="Candidate output directory")
    parser.add_argument("--out", required=True, help="Output prefix or .csv path")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline) / "localization_per_image.csv"
    candidate_path = Path(args.candidate) / "localization_per_image.csv"
    baseline = read_rows(baseline_path)
    candidate = read_rows(candidate_path)
    rows = paired_bootstrap(
        baseline, candidate, args.metrics, args.bootstrap_samples, args.seed
    )
    if not rows:
        raise RuntimeError("no finite metric pairs were found")
    output = Path(args.out)
    csv_path = output if output.suffix == ".csv" else output.with_suffix(".csv")
    json_path = csv_path.with_suffix(".json")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "results": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['metric']}: delta={row['delta']:.6f}, "
            f"paired_95CI=[{row['paired_ci95_low']:.6f}, {row['paired_ci95_high']:.6f}]"
        )
    print(f"CSV:  {csv_path}\nJSON: {json_path}")


if __name__ == "__main__":
    main()
