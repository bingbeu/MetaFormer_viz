"""Validate token scores against empirical second-order sensitivity on CUB.

The script does not retrain or alter the classifier.  For each selected test
image, it compares HVP, gradient, entropy, and distilled-student scores with a
central finite-difference estimate of the same compatibility objective.
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from vit_experiments.cub_dataset import CUB200
from vit_experiments.model import build_model
from vit_experiments.train import build_transforms


def parse_args():
    parser = argparse.ArgumentParser("Curv-Part sensitivity validation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--category-bank", default=None)
    parser.add_argument("--semantic-root", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fd-eps", type=float, default=0.1)
    parser.add_argument("--fd-samples", type=int, default=4)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    return parser.parse_args()


def stratified_indices(dataset, limit, seed):
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


def top_overlap(left, right, fraction):
    count = max(1, int(round(len(left) * fraction)))
    left_top = set(np.argpartition(left, -count)[-count:].tolist())
    right_top = set(np.argpartition(right, -count)[-count:].tolist())
    return len(left_top & right_top) / float(count)


def mean_sd(values):
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return {"n": 0, "mean": None, "sample_sd": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


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


def main():
    args = parse_args()
    if args.max_images < 1 or args.fd_eps <= 0 or args.fd_samples < 1:
        raise ValueError("max-images, fd-eps, and fd-samples must be positive")
    if not 0 < args.top_fraction <= 1:
        raise ValueError("top-fraction must be in (0, 1]")
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

    records = []
    processed = 0
    for images, _, semantics in loader:
        images = images.to(device, non_blocking=True)
        semantics = semantics.to(device, non_blocking=True)
        captured.clear()
        with torch.no_grad():
            model(images, semantics)

        batch_records = [defaultdict(list) for _ in range(images.shape[0])]
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
                finite_difference = arrays["finite_difference"][batch_index]
                for name in ("hvp", "gradient", "entropy", "student"):
                    score = arrays[name][batch_index]
                    batch_records[batch_index][
                        f"spearman_{name}_vs_finite_difference"
                    ].append(spearman(score, finite_difference))
                    batch_records[batch_index][
                        f"top_overlap_{name}_vs_finite_difference"
                    ].append(
                        top_overlap(score, finite_difference, args.top_fraction)
                    )
                batch_records[batch_index]["spearman_student_vs_hvp"].append(
                    spearman(
                        arrays["student"][batch_index],
                        arrays["hvp"][batch_index],
                    )
                )

        for batch_index, values in enumerate(batch_records):
            record = {"sample_index": indices[processed + batch_index]}
            for name, per_layer_values in values.items():
                record[name] = statistics.fmean(per_layer_values)
            records.append(record)
        processed += images.shape[0]

    for handle in handles:
        handle.remove()

    metric_names = sorted(name for name in records[0] if name != "sample_index")
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_images": len(records),
        "layers": list(model.generators.keys()),
        "finite_difference_eps": args.fd_eps,
        "finite_difference_samples": args.fd_samples,
        "top_fraction": args.top_fraction,
        "metrics": {
            name: mean_sd([record[name] for record in records])
            for name in metric_names
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_image": records}, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
