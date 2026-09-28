"""Validate token scores against equal-magnitude empirical perturbations.

This evaluation-only script compares HVP, distilled-student, first-order
gradient, entropy, Part attention, and deterministic random token rankings
against two held-out empirical targets computed on the same visual--semantic
compatibility objective:

1. symmetric absolute objective change under equal-L2 token perturbations;
2. absolute central second difference (empirical second-order sensitivity).

It reports per-image rank correlations, top-token overlap, top-k perturbation
response curves, bootstrap confidence intervals, and paired deltas against the
alternative rankings. No checkpoint selection or model update is performed.
"""

import argparse
import csv
import json
import random
import statistics
import zlib
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from vit_experiments.cub_dataset import CUB200
from vit_experiments.model import build_model
from vit_experiments.train import build_transforms


SCORE_NAMES = ("hvp", "student", "gradient", "entropy", "attention", "random")
TARGET_NAMES = ("perturbation", "finite_difference")


def parse_args():
    parser = argparse.ArgumentParser("Curv-Part perturbation sensitivity validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--category-bank", default=None)
    parser.add_argument("--semantic-root", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--fd-eps",
        type=float,
        default=0.1,
        help="Equal per-token L2 perturbation magnitude.",
    )
    parser.add_argument(
        "--fd-samples",
        type=int,
        default=8,
        help="Rademacher directions averaged for each empirical response.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.1,
        help="Fraction used for the single top-token overlap statistic.",
    )
    parser.add_argument(
        "--curve-fractions",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.2, 0.3),
        help="Token fractions used for top-k perturbation response curves.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def stratified_indices(dataset, limit, seed):
    """Round-robin class sampling with deterministic within-class shuffling."""
    by_class = defaultdict(list)
    for index, (_, target) in enumerate(dataset.samples):
        by_class[int(target)].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)
    selected = []
    depth = 0
    classes = sorted(by_class)
    while len(selected) < min(limit, len(dataset)):
        added = False
        for target in classes:
            if depth < len(by_class[target]):
                selected.append(by_class[target][depth])
                added = True
                if len(selected) == min(limit, len(dataset)):
                    break
        if not added:
            break
        depth += 1
    return selected


def rankdata(values):
    """Average ranks for ties without requiring SciPy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left, right):
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    if left_rank.std() == 0 or right_rank.std() == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def top_indices(values, fraction):
    count = max(1, int(round(len(values) * fraction)))
    return np.argpartition(values, -count)[-count:]


def top_overlap(left, right, fraction):
    left_top = set(top_indices(left, fraction).tolist())
    right_top = set(top_indices(right, fraction).tolist())
    return len(left_top & right_top) / float(len(left_top))


def top_response(score, empirical_response, fraction):
    """Actual perturbation response captured by a score's top-ranked tokens."""
    selected = top_indices(score, fraction)
    response = np.asarray(empirical_response, dtype=np.float64)
    total = float(response.sum())
    overall_mean = float(response.mean())
    selected_sum = float(response[selected].sum())
    selected_mean = float(response[selected].mean())
    return {
        "capture": selected_sum / total if total > 0 else float("nan"),
        "enrichment": (
            selected_mean / overall_mean if overall_mean > 0 else float("nan")
        ),
    }


def bootstrap_summary(values, samples, seed):
    values = np.asarray([value for value in values if np.isfinite(value)])
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "sample_sd": None,
            "bootstrap_ci95": [None, None],
        }
    mean = float(values.mean())
    sample_sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if samples < 1 or len(values) == 1:
        interval = [mean, mean]
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(values), size=(samples, len(values)))
        means = values[indices].mean(axis=1)
        interval = [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ]
    return {
        "n": int(len(values)),
        "mean": mean,
        "sample_sd": sample_sd,
        "bootstrap_ci95": interval,
    }


def metric_seed(base_seed, metric_name):
    return int(base_seed + zlib.crc32(metric_name.encode("utf-8")))


def load_experiment(checkpoint_path, overrides):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "args" not in checkpoint:
        raise KeyError("Checkpoint must contain the training 'args' dictionary")
    saved = dict(checkpoint["args"])
    if saved.get("model") != "curvpart_vit":
        raise ValueError("Sensitivity evaluation requires a Curv-Part ViT checkpoint")
    if overrides.category_bank is not None:
        saved["category_bank"] = overrides.category_bank
    if overrides.semantic_root is not None:
        saved["semantic_root"] = overrides.semantic_root
    model_args = SimpleNamespace(**saved)
    model = build_model(model_args)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, model_args


def deterministic_random_score(seed, sample_index, layer, length):
    layer_id = int(layer) if str(layer).isdigit() else zlib.crc32(str(layer).encode())
    local_seed = seed + 1000003 * int(sample_index) + 9176 * layer_id
    return np.random.default_rng(local_seed).random(length)


def fraction_tag(fraction):
    return f"{100.0 * fraction:g}pct".replace(".", "p")


def write_rows(path, rows):
    if not rows:
        return
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    """Replace non-finite scalars so strict JSON output never fails."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    args = parse_args()
    if args.max_images < 1 or args.fd_eps <= 0 or args.fd_samples < 1:
        raise ValueError("max-images, fd-eps, and fd-samples must be positive")
    fractions = [args.top_fraction, *args.curve_fractions]
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("top and curve fractions must be in (0, 1]")
    args.curve_fractions = tuple(sorted(set(args.curve_fractions)))
    if not torch.cuda.is_available():
        raise RuntimeError("Sensitivity evaluation requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    model, model_args = load_experiment(args.checkpoint, args)
    model = model.to(device).eval()
    _, test_transform = build_transforms(model_args.input_size)
    dataset = CUB200(
        root=args.data_path,
        train=False,
        transform=test_transform,
        semantic_root=getattr(model_args, "semantic_root", None),
        semantic_key=getattr(model_args, "semantic_key", "embedding_words"),
        semantic_dim=model_args.semantic_dim,
        max_semantic_tokens=model_args.max_semantic_tokens,
    )
    indices = stratified_indices(dataset, args.max_images, args.seed)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    captured = {}
    handles = []
    for layer, generator in model.generators.items():
        def capture(module, inputs, layer_name=layer):
            visual, conditions = inputs[:2]
            captured[layer_name] = (
                visual.detach(),
                [condition.detach() for condition in conditions],
            )
        handles.append(generator.register_forward_pre_hook(capture))

    per_image_layer = []
    processed = 0
    for images, _, semantics in loader:
        images = images.to(device, non_blocking=True)
        semantics = semantics.to(device, non_blocking=True)
        captured.clear()
        with torch.no_grad():
            model(images, semantics)

        for layer, generator in model.generators.items():
            visual, conditions = captured[layer]
            targets = generator.diagnostic_targets(
                visual,
                conditions,
                finite_difference_eps=args.fd_eps,
                finite_difference_samples=args.fd_samples,
            )
            arrays = {
                name: value.squeeze(-1).detach().float().cpu().numpy()
                for name, value in targets.items()
            }
            for batch_index in range(images.shape[0]):
                sample_index = indices[processed + batch_index]
                scores = {
                    name: arrays[name][batch_index]
                    for name in SCORE_NAMES
                    if name != "random"
                }
                scores["random"] = deterministic_random_score(
                    args.seed,
                    sample_index,
                    layer,
                    len(arrays[TARGET_NAMES[0]][batch_index]),
                )
                record = {"sample_index": sample_index, "layer": int(layer)}
                for target_name in TARGET_NAMES:
                    empirical = arrays[target_name][batch_index]
                    for score_name, score in scores.items():
                        stem = f"{score_name}_vs_{target_name}"
                        record[f"spearman_{stem}"] = spearman(score, empirical)
                        record[f"top_overlap_{stem}"] = top_overlap(
                            score, empirical, args.top_fraction
                        )
                        for fraction in args.curve_fractions:
                            response = top_response(score, empirical, fraction)
                            tag = fraction_tag(fraction)
                            record[f"capture_{stem}_{tag}"] = response["capture"]
                            record[f"enrichment_{stem}_{tag}"] = response[
                                "enrichment"
                            ]
                record["spearman_student_vs_hvp"] = spearman(
                    scores["student"], scores["hvp"]
                )
                per_image_layer.append(record)
        processed += images.shape[0]

    for handle in handles:
        handle.remove()

    # Average insertion layers within each image so images, not layers, remain
    # the independent units for uncertainty estimates.
    grouped = defaultdict(list)
    for record in per_image_layer:
        grouped[record["sample_index"]].append(record)
    per_image = []
    for sample_index, layer_records in sorted(grouped.items()):
        record = {"sample_index": sample_index}
        names = sorted(
            set().union(*(layer_record.keys() for layer_record in layer_records))
            - {"sample_index", "layer"}
        )
        for name in names:
            values = [layer_record[name] for layer_record in layer_records]
            finite = [value for value in values if np.isfinite(value)]
            record[name] = statistics.fmean(finite) if finite else float("nan")
        per_image.append(record)

    metric_names = sorted(name for name in per_image[0] if name != "sample_index")
    metrics = {
        name: bootstrap_summary(
            [record[name] for record in per_image],
            args.bootstrap_samples,
            metric_seed(args.seed, name),
        )
        for name in metric_names
    }

    # Paired per-image deltas avoid confounding by easy versus hard images.
    paired = {}
    for target_name in TARGET_NAMES:
        for metric_prefix in ("spearman", "top_overlap"):
            for candidate in ("hvp", "student"):
                candidate_key = f"{metric_prefix}_{candidate}_vs_{target_name}"
                for reference in ("gradient", "entropy", "attention", "random"):
                    reference_key = f"{metric_prefix}_{reference}_vs_{target_name}"
                    name = f"{candidate}_minus_{reference}_{metric_prefix}_vs_{target_name}"
                    paired[name] = bootstrap_summary(
                        [
                            record[candidate_key] - record[reference_key]
                            for record in per_image
                        ],
                        args.bootstrap_samples,
                        metric_seed(args.seed, name),
                    )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_images": len(per_image),
        "sampling": "class-stratified CUB test subset",
        "layers": list(model.generators.keys()),
        "perturbation_l2": args.fd_eps,
        "perturbation_directions": args.fd_samples,
        "top_fraction": args.top_fraction,
        "curve_fractions": list(args.curve_fractions),
        "bootstrap_samples": args.bootstrap_samples,
        "target_definitions": {
            "perturbation": (
                "symmetric absolute compatibility change per unit equal-L2 "
                "token perturbation"
            ),
            "finite_difference": (
                "absolute central second difference of token compatibility"
            ),
        },
        "score_definitions": {
            "hvp": "Hutchinson diagonal-Hessian L2 norm",
            "student": "distilled first-order curvature student",
            "gradient": "first-order compatibility-gradient L2 norm",
            "entropy": "per-token contribution to spatial compatibility entropy",
            "attention": "mean pre-dropout Part-query attention over image tokens",
            "random": "deterministic random token ranking",
        },
        "metrics": metrics,
        "paired_deltas": paired,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "per_image": per_image,
        "per_image_layer": per_image_layer,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)

    per_image_path = output.with_name(f"{output.stem}_per_image.csv")
    layer_path = output.with_name(f"{output.stem}_per_image_layer.csv")
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    write_rows(per_image_path, per_image)
    write_rows(layer_path, per_image_layer)
    write_rows(
        summary_path,
        [
            {
                "metric": name,
                "n": value["n"],
                "mean": value["mean"],
                "sample_sd": value["sample_sd"],
                "ci95_low": value["bootstrap_ci95"][0],
                "ci95_high": value["bootstrap_ci95"][1],
            }
            for name, value in {**metrics, **paired}.items()
        ],
    )

    concise = {
        "checkpoint": summary["checkpoint"],
        "num_images": summary["num_images"],
        "perturbation_l2": summary["perturbation_l2"],
        "perturbation_directions": summary["perturbation_directions"],
        "primary_metrics": {
            name: metrics[name]
            for name in (
                "spearman_hvp_vs_finite_difference",
                "spearman_student_vs_finite_difference",
                "spearman_gradient_vs_finite_difference",
                "spearman_entropy_vs_finite_difference",
                "spearman_attention_vs_finite_difference",
                "spearman_random_vs_finite_difference",
                "spearman_hvp_vs_perturbation",
                "spearman_gradient_vs_perturbation",
            )
        },
        "outputs": [
            str(output),
            str(summary_path),
            str(per_image_path),
            str(layer_path),
        ],
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
