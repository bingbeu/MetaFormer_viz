#!/usr/bin/env python3
"""Generate publication-ready Curv-Part evidence visualizations.

This script is intentionally isolated from the training/evaluation entry points.
It consumes the auxiliary tensors already returned by MetaFG_meta and never
changes model weights or the training graph.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

# Perceptually uniform, colour-blind-friendly maps. Never use jet/rainbow.
CMAP_ATTENTION = "magma"
CMAP_AGREEMENT = "viridis"
CMAP_DIFFERENCE = "cividis"
CMAP_MATRIX = "viridis"
ALLOWED_CMAPS = {
    CMAP_ATTENTION,
    CMAP_AGREEMENT,
    CMAP_DIFFERENCE,
    CMAP_MATRIX,
}

NUM_CLASSES = {
    "cub-200": 200,
    "nabirds": 555,
    "stanfordcars": 196,
    "aircraft": 100,
    "inaturelist2018": 8142,
    "inaturelist2021": 10000,
}


def configure_publication_style():
    """Use a compact white-background style suitable for CV papers."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def get_colormap(name):
    """Return a colormap on both recent and legacy Matplotlib versions."""
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    return plt.get_cmap(name)


def validate_colormaps():
    for name in ALLOWED_CMAPS:
        try:
            get_colormap(name)
        except ValueError as error:
            raise RuntimeError("Configured publication colormap is unavailable: {}".format(name)) from error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Curv-Part curvature, part evidence, and grounding."
    )
    parser.add_argument("--cfg", required=True, help="Model YAML configuration.")
    parser.add_argument("--checkpoint", required=True, help="Trained best.pth path.")
    parser.add_argument("--dataset", default="cub-200", choices=sorted(NUM_CLASSES))
    parser.add_argument("--data-root", default="/raid/datasets/cub-200")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--layer", type=int, default=2, choices=(1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="paper_visualization/outputs/sample_0000")
    parser.add_argument("--force-hvp", action="store_true")
    parser.add_argument("--hvp-layer", type=int, default=None, choices=(1, 2))
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--formats", nargs="+", default=("png", "pdf"), choices=("png", "pdf", "svg")
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # The plotting-only self-check intentionally works without PyTorch.
        pass


def load_config(cfg_path, dataset, img_size):
    from config import get_inference_config

    config = get_inference_config(SimpleNamespace(cfg=cfg_path))
    config.defrost()
    config.DATA.DATASET = dataset
    config.DATA.IMG_SIZE = int(img_size)
    config.DATA.ADD_META = True
    config.MODEL.NUM_CLASSES = NUM_CLASSES[dataset]
    config.freeze()
    return config


def load_model(config, checkpoint_path, device):
    import torch

    from models import build_model

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    incompatible = model.load_state_dict(state, strict=False)

    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    missing_part = [key for key in missing if "part_gen" in key]
    if missing_part:
        raise RuntimeError(
            "Checkpoint is missing Curv-Part parameters; first missing keys: "
            + ", ".join(missing_part[:8])
        )
    if missing or unexpected:
        print(
            "[checkpoint] non-strict load: missing={} unexpected={}".format(
                len(missing), len(unexpected)
            )
        )
    model.to(device)
    model.eval()
    return model


def load_cub_sample(config, data_root, sample_index, caption_index):
    import torch

    from data.build import build_transform
    from data.dataset_fg import DatasetMeta

    if config.DATA.DATASET != "cub-200":
        raise NotImplementedError(
            "The repository only exposes per-image text32 embeddings for CUB in "
            "DatasetMeta. Add a dataset-specific adapter before using another dataset."
        )

    transform = build_transform(is_train=False, config=config)
    dataset = DatasetMeta(
        root=data_root,
        transform=transform,
        train=False,
        aux_info=True,
        dataset="cub-200",
    )
    if not 0 <= sample_index < len(dataset):
        raise IndexError("sample-index {} outside [0, {})".format(sample_index, len(dataset)))

    image_path, target, all_meta = dataset.samples[sample_index]
    all_meta = np.asarray(all_meta)
    if all_meta.ndim < 2:
        raise ValueError("Expected multiple caption embeddings, got {}".format(all_meta.shape))
    if not 0 <= caption_index < all_meta.shape[0]:
        raise IndexError(
            "caption-index {} outside [0, {})".format(caption_index, all_meta.shape[0])
        )

    raw_image = Image.open(image_path).convert("RGB")
    image_tensor = transform(raw_image).unsqueeze(0)
    meta_tensor = torch.as_tensor(all_meta[caption_index], dtype=torch.float32).unsqueeze(0)

    caption = ""
    if sample_index < len(dataset.images_info):
        text_list = dataset.images_info[sample_index].get("text_list", [])
        if caption_index < len(text_list):
            caption = text_list[caption_index].strip()

    class_name = read_cub_class_name(data_root, int(target))
    return image_tensor, meta_tensor, int(target), image_path, caption, class_name


def read_cub_class_name(data_root, target):
    classes_path = Path(data_root) / "CUB_200_2011" / "classes.txt"
    if not classes_path.exists():
        return "class-{:03d}".format(target + 1)
    with classes_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip().split(maxsplit=1) for line in handle if line.strip()]
    if 0 <= target < len(lines):
        return lines[target][1].replace("_", " ")
    return "class-{:03d}".format(target + 1)


def tensor_to_display_image(image_tensor):
    image = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
    image = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(image, 0.0, 1.0)


def as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def vector_to_grid(value):
    value = np.asarray(value).squeeze()
    if value.ndim != 1:
        raise ValueError("Expected token vector, got shape {}".format(value.shape))
    side = int(round(math.sqrt(value.size)))
    if side * side != value.size:
        raise ValueError("Token count {} is not a square grid".format(value.size))
    return value.reshape(side, side)


def part_tensor_to_grids(value):
    value = np.asarray(value).squeeze()
    if value.ndim != 2:
        raise ValueError("Expected [parts, tokens], got shape {}".format(value.shape))
    side = int(round(math.sqrt(value.shape[1])))
    if side * side != value.shape[1]:
        raise ValueError("Token count {} is not a square grid".format(value.shape[1]))
    return value.reshape(value.shape[0], side, side)


def percentile_rank(values):
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(flat.size, dtype=np.float64)
    if flat.size > 1:
        ranks /= float(flat.size - 1)
    return ranks.reshape(np.asarray(values).shape)


def resize_heatmap(heatmap, height, width):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    pil_heatmap = Image.fromarray(heatmap, mode="F")
    resized = pil_heatmap.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def robust_limits(arrays, lower=1.0, upper=99.0):
    flat = np.concatenate([np.asarray(array).reshape(-1) for array in arrays])
    vmin, vmax = np.percentile(flat, [lower, upper])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(flat))
        vmax = float(np.nanmax(flat))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    return float(vmin), float(vmax)


def overlay_heatmap(image, heatmap, cmap, alpha, vmin, vmax):
    height, width = image.shape[:2]
    heatmap = resize_heatmap(heatmap, height, width)
    normalized = np.clip((heatmap - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    colour = get_colormap(cmap)(normalized)[..., :3]
    # Heat-dependent transparency preserves the underlying image in low-response
    # regions and avoids painting the complete background with the colormap.
    blend = (float(alpha) * normalized)[..., None]
    return np.clip((1.0 - blend) * image + blend * colour, 0.0, 1.0)


def compute_part_statistics(part_grids, top_fraction=0.10):
    parts = np.asarray(part_grids, dtype=np.float64)
    p, h, w = parts.shape
    flat = np.maximum(parts.reshape(p, -1), 0.0)
    flat /= np.maximum(flat.sum(axis=1, keepdims=True), 1e-12)

    overlap = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            # Distribution overlap coefficient: 0=no shared mass, 1=identical.
            overlap[i, j] = np.minimum(flat[i], flat[j]).sum()

    top_k = max(1, int(round(flat.shape[1] * float(top_fraction))))
    top_masks = np.zeros_like(flat, dtype=np.float64)
    for i in range(p):
        indices = np.argpartition(flat[i], -top_k)[-top_k:]
        top_masks[i, indices] = 1.0

    agreement = top_masks.mean(axis=0).reshape(h, w)
    coverage = flat.max(axis=0).reshape(h, w)
    consensus = flat.mean(axis=0).reshape(h, w)
    peaks = [np.unravel_index(int(np.argmax(row)), (h, w)) for row in flat]

    off_diag = overlap[~np.eye(p, dtype=bool)] if p > 1 else np.asarray([1.0])
    metrics = {
        "num_parts": int(p),
        "token_grid": [int(h), int(w)],
        "top_fraction": float(top_fraction),
        "mean_pairwise_overlap": float(off_diag.mean()),
        "max_pairwise_overlap": float(off_diag.max()),
        "min_pairwise_overlap": float(off_diag.min()),
        "peak_locations_yx": [[int(y), int(x)] for y, x in peaks],
        "max_part_agreement": float(agreement.max()),
    }
    return overlap, agreement, coverage, consensus, metrics


def save_figure(fig, output_stem, formats):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(str(output_stem) + "." + extension, **kwargs)
    plt.close(fig)


def plot_part_evidence(image, part_grids, output_stem, formats, alpha, top_fraction):
    overlap, agreement, coverage, consensus, metrics = compute_part_statistics(
        part_grids, top_fraction=top_fraction
    )
    vmin, vmax = robust_limits(list(part_grids))

    panels = [("Input", image)]
    panels.append(
        (
            "Part agreement",
            overlay_heatmap(image, agreement, CMAP_AGREEMENT, alpha, 0.0, 1.0),
        )
    )
    coverage_limits = robust_limits([coverage])
    panels.append(
        (
            "Part coverage",
            overlay_heatmap(
                image, coverage, CMAP_ATTENTION, alpha, coverage_limits[0], coverage_limits[1]
            ),
        )
    )
    for index, part_map in enumerate(part_grids):
        panels.append(
            (
                "Part {}".format(index + 1),
                overlay_heatmap(image, part_map, CMAP_ATTENTION, alpha, vmin, vmax),
            )
        )

    columns = 4
    rows = int(math.ceil(len(panels) / float(columns)))
    fig, axes = plt.subplots(rows, columns, figsize=(7.16, 1.78 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(panels) :]:
        axis.axis("off")
    save_figure(fig, str(output_stem) + "_part_attention", formats)

    fig, axis = plt.subplots(figsize=(3.45, 3.0), constrained_layout=True)
    image_handle = axis.imshow(overlap, cmap=CMAP_MATRIX, vmin=0.0, vmax=1.0)
    axis.set_xlabel("Part index")
    axis.set_ylabel("Part index")
    axis.set_xticks(np.arange(overlap.shape[0]))
    axis.set_yticks(np.arange(overlap.shape[0]))
    axis.set_xticklabels(np.arange(1, overlap.shape[0] + 1))
    axis.set_yticklabels(np.arange(1, overlap.shape[0] + 1))
    colourbar = fig.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04)
    colourbar.set_label("Attention-mass overlap")
    axis.set_title("Part evidence agreement")
    save_figure(fig, str(output_stem) + "_part_overlap", formats)
    return metrics, overlap, agreement, coverage, consensus


def plot_teacher_student(image, student_grid, teacher_grid, output_stem, formats, alpha):
    student_rank = percentile_rank(student_grid)
    teacher_rank = percentile_rank(teacher_grid)
    difference = np.abs(student_rank - teacher_rank)
    rho = float(np.corrcoef(student_rank.reshape(-1), teacher_rank.reshape(-1))[0, 1])

    panels = [
        ("Input", image),
        (
            "Student percentile",
            overlay_heatmap(image, student_rank, CMAP_ATTENTION, alpha, 0.0, 1.0),
        ),
        (
            "HVP teacher percentile",
            overlay_heatmap(image, teacher_rank, CMAP_ATTENTION, alpha, 0.0, 1.0),
        ),
        (
            "Absolute rank difference\nSpearman = {:.3f}".format(rho),
            overlay_heatmap(image, difference, CMAP_DIFFERENCE, alpha, 0.0, 1.0),
        ),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0), constrained_layout=True)
    for axis, (title, panel) in zip(axes, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    save_figure(fig, str(output_stem) + "_teacher_student", formats)
    return {
        "teacher_student_spearman": rho,
        "mean_absolute_rank_difference": float(difference.mean()),
    }


def make_token_labels(caption, token_count):
    # Avoid online model downloads. The caption is saved verbatim in metrics.json;
    # position labels remain valid even when the tokenizer cache is unavailable.
    labels = ["T{:02d}".format(index + 1) for index in range(token_count)]
    return labels


def plot_semantic_grounding(attr_attn, caption, output_stem, formats):
    attr_attn = np.asarray(attr_attn).squeeze()
    if attr_attn.ndim != 2:
        raise ValueError("Expected semantic attention [parts, text tokens]")
    labels = make_token_labels(caption, attr_attn.shape[1])
    fig, axis = plt.subplots(figsize=(7.16, 2.7), constrained_layout=True)
    handle = axis.imshow(attr_attn, aspect="auto", cmap=CMAP_AGREEMENT)
    axis.set_xlabel("Description-token position")
    axis.set_ylabel("Part index")
    axis.set_yticks(np.arange(attr_attn.shape[0]))
    axis.set_yticklabels(np.arange(1, attr_attn.shape[0] + 1))
    step = max(1, int(math.ceil(attr_attn.shape[1] / 16.0)))
    ticks = np.arange(0, attr_attn.shape[1], step)
    axis.set_xticks(ticks)
    axis.set_xticklabels([labels[index] for index in ticks], rotation=45, ha="right")
    colourbar = fig.colorbar(handle, ax=axis, fraction=0.025, pad=0.02)
    colourbar.set_label("Grounding weight")
    title = "Part-to-description grounding"
    if caption:
        shortened = caption if len(caption) <= 110 else caption[:107] + "..."
        title += "\n" + shortened
    axis.set_title(title)
    save_figure(fig, str(output_stem) + "_semantic_grounding", formats)


def plot_overview(
    image,
    student_grid,
    teacher_grid,
    agreement,
    coverage,
    output_stem,
    formats,
    alpha,
):
    student_rank = percentile_rank(student_grid)
    panels = [
        ("Input", image),
        (
            "Student curvature",
            overlay_heatmap(image, student_rank, CMAP_ATTENTION, alpha, 0.0, 1.0),
        ),
    ]
    if teacher_grid is not None:
        teacher_rank = percentile_rank(teacher_grid)
        panels.append(
            (
                "HVP teacher",
                overlay_heatmap(image, teacher_rank, CMAP_ATTENTION, alpha, 0.0, 1.0),
            )
        )
    panels.extend(
        [
            (
                "Part agreement",
                overlay_heatmap(image, agreement, CMAP_AGREEMENT, alpha, 0.0, 1.0),
            ),
            (
                "Part coverage",
                overlay_heatmap(
                    image,
                    coverage,
                    CMAP_ATTENTION,
                    alpha,
                    robust_limits([coverage])[0],
                    robust_limits([coverage])[1],
                ),
            ),
        ]
    )
    fig, axes = plt.subplots(1, len(panels), figsize=(7.16, 1.65), constrained_layout=True)
    for axis, (title, panel) in zip(np.asarray(axes).reshape(-1), panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    save_figure(fig, str(output_stem) + "_overview", formats)


def run_visualization(args):
    import torch

    configure_publication_style()
    seed_everything(args.seed)
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top-fraction must be in (0, 1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay-alpha must be in [0, 1]")
    validate_colormaps()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    config = load_config(args.cfg, args.dataset, args.img_size)
    image_tensor, meta_tensor, target, image_path, caption, class_name = load_cub_sample(
        config,
        args.data_root,
        args.sample_index,
        args.caption_index,
    )
    model = load_model(config, args.checkpoint, device)
    image_tensor = image_tensor.to(device)
    meta_tensor = meta_tensor.to(device)

    if args.force_hvp:
        with torch.enable_grad():
            logits, aux = model(
                image_tensor,
                meta_tensor,
                return_aux=True,
                force_hvp=True,
                force_hvp_layer=args.hvp_layer or args.layer,
            )
    else:
        with torch.no_grad():
            logits, aux = model(image_tensor, meta_tensor, return_aux=True)

    probabilities = torch.softmax(logits.detach(), dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    prediction = int(prediction.item())
    confidence = float(confidence.item())

    suffix = str(args.layer)
    part_value = aux.get("part_attn_" + suffix)
    student_value = aux.get("student_curvature_" + suffix, aux.get("curvature_" + suffix))
    teacher_value = aux.get("hvp_curvature_" + suffix)
    attr_value = aux.get("attr_attn_" + suffix)
    if part_value is None:
        raise RuntimeError("part_attn_{} was not returned by the model".format(suffix))
    if student_value is None:
        raise RuntimeError("student curvature for layer {} was not returned".format(suffix))
    if args.force_hvp and teacher_value is None:
        raise RuntimeError(
            "force-hvp was requested, but hvp_curvature_{} is None".format(suffix)
        )

    part_grids = part_tensor_to_grids(as_numpy(part_value)[0])
    student_grid = vector_to_grid(as_numpy(student_value)[0])
    teacher_grid = None
    if teacher_value is not None:
        teacher_grid = vector_to_grid(as_numpy(teacher_value)[0])
    image = tensor_to_display_image(image_tensor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "layer{}".format(args.layer)

    metrics, overlap, agreement, coverage, consensus = plot_part_evidence(
        image,
        part_grids,
        stem,
        args.formats,
        args.overlay_alpha,
        args.top_fraction,
    )
    if teacher_grid is not None:
        metrics.update(
            plot_teacher_student(
                image,
                student_grid,
                teacher_grid,
                stem,
                args.formats,
                args.overlay_alpha,
            )
        )
    if attr_value is not None:
        plot_semantic_grounding(as_numpy(attr_value)[0], caption, stem, args.formats)
    plot_overview(
        image,
        student_grid,
        teacher_grid,
        agreement,
        coverage,
        stem,
        args.formats,
        args.overlay_alpha,
    )

    saved_maps = {
        "part_attention": part_grids,
        "part_overlap": overlap,
        "part_agreement": agreement,
        "part_coverage": coverage,
        "part_consensus": consensus,
        "student_curvature": student_grid,
    }
    if teacher_grid is not None:
        saved_maps["teacher_curvature"] = teacher_grid
    np.savez_compressed(
        output_dir / "layer{}_maps.npz".format(args.layer), **saved_maps
    )
    metrics.update(
        {
            "sample_index": int(args.sample_index),
            "caption_index": int(args.caption_index),
            "image_path": str(image_path),
            "target": int(target),
            "target_name": class_name,
            "prediction": prediction,
            "confidence": confidence,
            "correct": bool(prediction == target),
            "layer": int(args.layer),
            "caption": caption,
            "checkpoint": str(args.checkpoint),
            "colormaps": {
                "attention": CMAP_ATTENTION,
                "agreement": CMAP_AGREEMENT,
                "difference": CMAP_DIFFERENCE,
                "matrix": CMAP_MATRIX,
            },
        }
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print("[done] outputs: {}".format(output_dir.resolve()))
    print(
        "[result] target={} prediction={} confidence={:.4f} mean_part_overlap={:.4f}".format(
            target, prediction, confidence, metrics["mean_pairwise_overlap"]
        )
    )


def main():
    run_visualization(parse_args())


if __name__ == "__main__":
    main()
