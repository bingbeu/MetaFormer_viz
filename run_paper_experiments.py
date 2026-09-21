#!/usr/bin/env python3
"""Run reproducible, checkpoint-only localization experiments for Curv-Part.

This driver intentionally delegates every measurement to
``evaluate_localization.py``.  It only fixes experiment grids, records exact
commands, skips completed runs, and creates paired comparisons.  Therefore it
does not change the model or the definition of any localization metric.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Run:
    name: str
    layer: int = 2
    mode: str = "raw"
    top_fraction: float = 0.20
    extra: tuple[str, ...] = field(default_factory=tuple)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-root", default="output/paper_experiments")
    parser.add_argument(
        "--suite", nargs="+", default=["core"],
        choices=("core", "layers", "thresholds", "causal", "diagnostics", "all"),
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 means full CUB test set")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-attention-maps", type=int, default=20)
    parser.add_argument("--include-hvp", action="store_true", help="Add slow HVP teacher maps")
    parser.add_argument("--force", action="store_true", help="Rerun completed output directories")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def experiment_grid(suites: list[str]) -> list[Run]:
    selected = set(suites)
    if "all" in selected:
        selected = {"core", "layers", "thresholds", "causal", "diagnostics"}
    runs: list[Run] = []
    if "core" in selected:
        runs += [
            Run("raw_layer2", mode="raw"),
            Run("cosine_mean_norm_layer2", mode="cosine_mean_norm"),
            Run("cosine_rms_layer2", mode="cosine_rms"),
        ]
    if "layers" in selected:
        runs += [
            Run("raw_layer1", layer=1, mode="raw"),
            Run("cosine_rms_layer1", layer=1, mode="cosine_rms"),
            Run("raw_layer2", layer=2, mode="raw"),
            Run("cosine_rms_layer2", layer=2, mode="cosine_rms"),
        ]
    if "thresholds" in selected:
        runs += [
            Run(f"cosine_rms_layer2_top{int(frac * 100):02d}", mode="cosine_rms", top_fraction=frac)
            for frac in (0.10, 0.20, 0.30)
        ]
    if "causal" in selected:
        runs.append(Run(
            "cosine_rms_layer2_causal", mode="cosine_rms",
            extra=("--causal-deletion", "--deletion-size", "0.15",
                   "--deletion-random-samples", "5", "--deletion-batch-size", "4",
                   "--flip-stability"),
        ))
    if "diagnostics" in selected:
        runs.append(Run(
            "raw_layer2_diagnostics", mode="raw",
            extra=("--attention-decomposition", "--content-norm-diagnostic",
                   "--save-decomposition-maps", "20", "--save-content-norm-maps", "20"),
        ))

    unique: dict[str, Run] = {}
    for run in runs:
        unique.setdefault(run.name, run)
    return list(unique.values())


def command_for(args: argparse.Namespace, run: Run) -> list[str]:
    sources = ["curvature", "curv_weight", "part_attention"]
    if args.include_hvp:
        sources.append("hvp_curvature")
    command = [
        args.python, "evaluate_localization.py",
        "--cfg", args.cfg,
        "--ckpt", args.ckpt,
        "--out", str(Path(args.out_root) / run.name),
        "--layer", str(run.layer),
        "--map-sources", *sources,
        "--batch-size", str(1 if "--causal-deletion" in run.extra else args.batch_size),
        "--num-workers", str(args.num_workers),
        "--max-images", str(args.max_images),
        "--sample-mode", "stratified",
        "--content-attention-mode", run.mode,
        "--top-fraction", str(run.top_fraction),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--seed", str(args.seed),
        "--save-overlays", "0",
        "--save-attention-maps", str(args.save_attention_maps),
        "--visualize-top-k", "4",
        "--visualize-token-selection", "fixed",
        "--attention-aggregation", "mean",
        "--attention-interpolation", "nearest",
        *run.extra,
    ]
    return command


def main() -> None:
    args = parse_args()
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    runs = experiment_grid(args.suite)
    manifest = {
        "cfg": args.cfg,
        "ckpt": args.ckpt,
        "suite": args.suite,
        "runs": [],
    }
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu

    for run in runs:
        output = root / run.name
        summary = output / "localization_summary.json"
        command = command_for(args, run)
        rendered = shlex.join(command)
        manifest["runs"].append({"name": run.name, "command": rendered})
        print(f"\n[{run.name}] {rendered}", flush=True)
        if summary.exists() and not args.force:
            print(f"[{run.name}] complete; use --force to rerun", flush=True)
            continue
        if not args.dry_run:
            output.mkdir(parents=True, exist_ok=True)
            (output / "command.txt").write_text(rendered + "\n", encoding="utf-8")
            subprocess.run(command, check=True, env=environment)

    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nManifest: {root / 'manifest.json'}")
    print("Compare paired runs with compare_localization_runs.py; see PAPER_EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
