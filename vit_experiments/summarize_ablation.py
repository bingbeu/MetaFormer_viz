"""Summarize fixed-final multi-seed ViT ablations."""

import argparse
import csv
import json
import statistics
from pathlib import Path


DEFAULT_MODES = (
    "baseline",
    "part_control",
    "gradient_teacher",
    "entropy_teacher",
    "full",
)


def parse_args():
    parser = argparse.ArgumentParser("Summarize ViT ablation matrix")
    parser.add_argument("--root", default="outputs/vit_controls")
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def final_metrics(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if "acc1" in record:
                records.append(record)
    if not records:
        raise RuntimeError(f"No evaluated epoch in {path}")
    return max(records, key=lambda record: record["epoch"])


def mean_sd(values):
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main():
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / "ablation_summary.csv"
    by_mode = {}
    rows = []
    for mode in args.modes:
        values = {}
        for seed in args.seeds:
            record = final_metrics(root / f"{mode}_seed{seed}" / "log.jsonl")
            values[seed] = float(record["acc1"])
        by_mode[mode] = values
        mean, sample_sd = mean_sd(list(values.values()))
        rows.append(
            {
                "mode": mode,
                "n": len(values),
                "mean_acc1": mean,
                "sample_sd": sample_sd,
                "paired_delta_vs_baseline": "",
                **{f"seed_{seed}": values[seed] for seed in args.seeds},
            }
        )

    baseline = by_mode.get("baseline")
    if baseline is not None:
        for row in rows:
            mode = row["mode"]
            deltas = [by_mode[mode][seed] - baseline[seed] for seed in args.seeds]
            delta_mean, delta_sd = mean_sd(deltas)
            row["paired_delta_vs_baseline"] = f"{delta_mean:.4f} +/- {delta_sd:.4f}"

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['mode']:>18}: {row['mean_acc1']:.2f} +/- "
            f"{row['sample_sd']:.2f}; delta={row['paired_delta_vs_baseline']}"
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
