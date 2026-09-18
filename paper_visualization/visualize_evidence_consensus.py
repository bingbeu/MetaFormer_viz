#!/usr/bin/env python3
"""Paper-facing Curv-Part visual analysis.

This entry point deliberately excludes the single-image Teacher--Student panel
and the position-only semantic heatmap.  Teacher--Student fidelity is a
dataset-level claim and should be reported with the existing aggregate
fidelity figure, not inferred from a stochastic HVP draw for one image.

The plots here answer a different question: where does the deployed student
place curvature-guided evidence, and do the learned part queries agree or
specialise?  Every attention map is visualised as *excess mass above uniform*
so that a nearly uniform softmax cannot look artificially salient.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_visualization.visualize_part_evidence import (
    as_numpy,
    configure_publication_style,
    load_config,
    load_cub_sample,
    load_model,
    part_tensor_to_grids,
    read_cub_class_name,
    resize_heatmap,
    seed_everything,
    tensor_to_display_image,
    vector_to_grid,
)


CMAP_EVIDENCE = "inferno"
CMAP_AGREEMENT = "viridis"
CMAP_DIVERGENCE = "cividis"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate claim-aligned Curv-Part visual analysis."
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="cub-200", choices=("cub-200",))
    parser.add_argument("--data-root", default="/raid/datasets/cub-200")
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=(0,),
        help="Pre-declared test indices. Use several fixed indices for a paper gallery.",
    )
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--layer", type=int, default=2, choices=(1, 2))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", default="paper_visualization/outputs/evidence_consensus"
    )
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--overlay-alpha", type=float, default=0.72)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--formats", nargs="+", default=("png", "pdf"), choices=("png", "pdf", "svg")
    )
    return parser.parse_args()


def _cmap(name):
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    return plt.get_cmap(name)


def save_figure(fig, stem, formats):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 300
        fig.savefig(str(stem) + "." + extension, **kwargs)
    plt.close(fig)


def robust_unit_map(values, baseline=None, upper_percentile=99.0):
    """Map non-negative evidence to [0, 1] without inventing ranks for ties."""
    array = np.asarray(values, dtype=np.float64)
    if baseline is None:
        baseline = float(np.nanmin(array))
    evidence = np.maximum(array - float(baseline), 0.0)
    positive = evidence[evidence > 0]
    if positive.size == 0:
        return np.zeros_like(evidence, dtype=np.float64)
    scale = float(np.percentile(positive, upper_percentile))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(positive.max())
    return np.clip(evidence / max(scale, 1e-12), 0.0, 1.0)


def attention_excess(part_grids):
    """Return attention mass above the uniform 1/N reference."""
    parts = np.asarray(part_grids, dtype=np.float64)
    if parts.ndim != 3:
        raise ValueError("Expected [parts, height, width]")
    flat = np.maximum(parts.reshape(parts.shape[0], -1), 0.0)
    row_sums = flat.sum(axis=1, keepdims=True)
    flat = flat / np.maximum(row_sums, 1e-12)
    uniform = 1.0 / float(flat.shape[1])
    excess = np.maximum(flat - uniform, 0.0)
    return excess.reshape(parts.shape), flat.reshape(parts.shape), uniform


def compute_query_statistics(part_grids, top_fraction):
    excess, normalized, uniform = attention_excess(part_grids)
    p, h, w = normalized.shape
    flat = normalized.reshape(p, -1)

    overlap = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(p):
            overlap[i, j] = np.minimum(flat[i], flat[j]).sum()

    top_k = max(1, int(round(flat.shape[1] * float(top_fraction))))
    masks = np.zeros_like(flat, dtype=np.float64)
    for index in range(p):
        selected = np.argpartition(flat[index], -top_k)[-top_k:]
        masks[index, selected] = 1.0
    agreement = masks.mean(axis=0).reshape(h, w)
    consensus = excess.mean(axis=0)
    peaks = [np.unravel_index(int(np.argmax(row)), (h, w)) for row in flat]
    off_diag = overlap[~np.eye(p, dtype=bool)] if p > 1 else np.asarray([1.0])

    flat_student_reference = None
    metrics = {
        "num_parts": int(p),
        "token_grid": [int(h), int(w)],
        "uniform_attention": float(uniform),
        "top_fraction": float(top_fraction),
        "mean_pairwise_overlap": float(off_diag.mean()),
        "min_pairwise_overlap": float(off_diag.min()),
        "max_pairwise_overlap": float(off_diag.max()),
        "mean_pairwise_divergence": float((1.0 - off_diag).mean()),
        "peak_locations_yx": [[int(y), int(x)] for y, x in peaks],
        "all_peaks_identical": bool(len(set(peaks)) == 1),
    }
    if metrics["mean_pairwise_overlap"] >= 0.98:
        metrics["interpretation_status"] = "high_query_redundancy"
        metrics["allowed_claim"] = "shared_evidence_consensus_only"
        metrics["forbidden_claim"] = "distinct_or_complementary_parts"
    else:
        metrics["interpretation_status"] = "mixed_consensus_and_specialisation"
        metrics["allowed_claim"] = "inspect_per-query_maps_before_claiming_specialisation"
        metrics["forbidden_claim"] = "none_without_dataset_level_validation"
    return excess, consensus, agreement, overlap, metrics


def overlay_evidence(image, unit_map, cmap_name, alpha):
    heat = resize_heatmap(unit_map, image.shape[0], image.shape[1])
    heat = np.clip(heat, 0.0, 1.0)
    colour = _cmap(cmap_name)(heat)[..., :3]
    # Power correction keeps weak numerical noise transparent while retaining
    # compact high-response evidence.
    blend = (float(alpha) * np.power(heat, 0.75))[..., None]
    return np.clip((1.0 - blend) * image + blend * colour, 0.0, 1.0)


def prepare_record(image, student_grid, part_grids, metadata, top_fraction):
    excess, consensus, agreement, overlap, statistics = compute_query_statistics(
        part_grids, top_fraction
    )
    student_unit = robust_unit_map(student_grid)
    consensus_unit = robust_unit_map(consensus, baseline=0.0)
    # One shared scale across all queries prevents per-panel normalization from
    # exaggerating weak queries or manufacturing apparent diversity.
    positive_part = excess[excess > 0]
    if positive_part.size:
        shared_part_scale = float(np.percentile(positive_part, 99.0))
        shared_part_scale = max(shared_part_scale, 1e-12)
        part_unit = np.clip(excess / shared_part_scale, 0.0, 1.0)
    else:
        shared_part_scale = 0.0
        part_unit = np.zeros_like(excess, dtype=np.float64)
    student_flat = np.asarray(student_grid).reshape(-1)
    floor = float(student_flat.min())
    statistics["student_floor_fraction"] = float(
        np.mean(np.isclose(student_flat, floor, rtol=0.0, atol=1e-8))
    )
    statistics["shared_part_scale_99th"] = float(shared_part_scale)
    statistics.update(metadata)
    return {
        "image": image,
        "student_unit": student_unit,
        "consensus_unit": consensus_unit,
        "agreement": agreement,
        "part_unit": part_unit,
        "overlap": overlap,
        "metrics": statistics,
        "part_raw": np.asarray(part_grids),
        "student_raw": np.asarray(student_grid),
    }


def plot_gallery(records, output_stem, formats, alpha, layer):
    rows = len(records)
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(7.16, max(1.72 * rows, 1.9)),
        constrained_layout=True,
        squeeze=False,
    )
    titles = [
        "(a) Input",
        "(b) Student importance",
        "(c) Shared part evidence",
        "(d) Query agreement",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="semibold")

    for row, record in enumerate(records):
        image = record["image"]
        panels = [
            image,
            overlay_evidence(image, record["student_unit"], CMAP_EVIDENCE, alpha),
            overlay_evidence(image, record["consensus_unit"], CMAP_EVIDENCE, alpha),
            overlay_evidence(image, record["agreement"], CMAP_AGREEMENT, alpha),
        ]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
        m = record["metrics"]
        marker = "correct" if m["correct"] else "error"
        axes[row, 0].text(
            0.02,
            0.03,
            "#{:04d} | {}".format(m["sample_index"], marker),
            transform=axes[row, 0].transAxes,
            fontsize=6.5,
            color="black",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.88, "linewidth": 0},
        )

    evidence_bar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP_EVIDENCE)
    agreement_bar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP_AGREEMENT)
    cb1 = fig.colorbar(
        evidence_bar,
        ax=axes[:, 1:3].ravel().tolist(),
        orientation="horizontal",
        fraction=0.025,
        pad=0.015,
        aspect=50,
    )
    cb1.set_label("Within-image normalized evidence", labelpad=2)
    cb2 = fig.colorbar(
        agreement_bar,
        ax=axes[:, 3].ravel().tolist(),
        orientation="horizontal",
        fraction=0.025,
        pad=0.015,
        aspect=25,
    )
    cb2.set_label("Fraction of agreeing queries", labelpad=2)
    save_figure(fig, output_stem, formats)


def plot_query_diagnostics(record, output_stem, formats, alpha):
    parts = record["part_unit"]
    columns = 4
    rows = int(math.ceil(parts.shape[0] / float(columns)))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.16, 1.82 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)
    for index, (axis, unit_map) in enumerate(zip(axes, parts)):
        axis.imshow(overlay_evidence(record["image"], unit_map, CMAP_EVIDENCE, alpha))
        axis.set_title("P{}".format(index + 1), fontweight="semibold")
        axis.axis("off")
    for axis in axes[parts.shape[0] :]:
        axis.axis("off")
    bar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP_EVIDENCE)
    cb = fig.colorbar(
        bar,
        ax=axes[: parts.shape[0]].tolist(),
        orientation="horizontal",
        fraction=0.025,
        pad=0.02,
        aspect=50,
    )
    cb.set_label("Excess attention above uniform (normalized)", labelpad=2)
    save_figure(fig, str(output_stem) + "_part_queries", formats)

    divergence = 1.0 - record["overlap"]
    masked = np.ma.array(divergence, mask=np.eye(divergence.shape[0], dtype=bool))
    vmax = float(masked.max()) if masked.count() else 1.0
    vmax = max(vmax, 1e-6)
    fig, axis = plt.subplots(figsize=(3.45, 3.0), constrained_layout=True)
    handle = axis.imshow(masked, cmap=CMAP_DIVERGENCE, vmin=0.0, vmax=vmax)
    axis.set_xlabel("Part-query index")
    axis.set_ylabel("Part-query index")
    ticks = np.arange(divergence.shape[0])
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    axis.set_xticklabels(ticks + 1)
    axis.set_yticklabels(ticks + 1)
    cb = fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)
    cb.set_label("Divergence (1 - mass overlap)")
    save_figure(fig, str(output_stem) + "_query_divergence", formats)


def validate_args(args):
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if len(set(args.sample_indices)) != len(args.sample_indices):
        raise ValueError("--sample-indices contains duplicates")
    for name in (CMAP_EVIDENCE, CMAP_AGREEMENT, CMAP_DIVERGENCE):
        _cmap(name)


def run(args):
    import torch

    configure_publication_style()
    seed_everything(args.seed)
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config = load_config(args.cfg, args.dataset, args.img_size)
    model = load_model(config, args.checkpoint, device)
    records = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index in args.sample_indices:
        image_tensor, meta_tensor, target, image_path, caption, target_name = load_cub_sample(
            config,
            args.data_root,
            sample_index,
            args.caption_index,
        )
        image_tensor = image_tensor.to(device)
        meta_tensor = meta_tensor.to(device)
        with torch.no_grad():
            logits, aux = model(image_tensor, meta_tensor, return_aux=True)

        probabilities = torch.softmax(logits, dim=-1)
        confidence, prediction = probabilities.max(dim=-1)
        prediction = int(prediction.item())
        suffix = str(args.layer)
        part_value = aux.get("part_attn_" + suffix)
        student_value = aux.get(
            "student_curvature_" + suffix, aux.get("curvature_" + suffix)
        )
        if part_value is None or student_value is None:
            raise RuntimeError(
                "Model did not return part attention/student curvature for layer {}".format(args.layer)
            )
        part_grids = part_tensor_to_grids(as_numpy(part_value)[0])
        student_grid = vector_to_grid(as_numpy(student_value)[0])
        if part_grids.shape[1:] != student_grid.shape:
            raise RuntimeError("Part and student token grids do not match")
        attention_sums = part_grids.reshape(part_grids.shape[0], -1).sum(axis=1)
        if not np.allclose(attention_sums, 1.0, rtol=2e-3, atol=2e-3):
            raise RuntimeError(
                "part_attn rows are not spatial softmax distributions: {}".format(attention_sums)
            )

        metadata = {
            "sample_index": int(sample_index),
            "caption_index": int(args.caption_index),
            "image_path": str(image_path),
            "caption": caption,
            "target": int(target),
            "target_name": target_name,
            "prediction": prediction,
            "prediction_name": read_cub_class_name(args.data_root, prediction),
            "confidence": float(confidence.item()),
            "correct": bool(prediction == target),
            "layer": int(args.layer),
        }
        record = prepare_record(
            tensor_to_display_image(image_tensor),
            student_grid,
            part_grids,
            metadata,
            args.top_fraction,
        )
        records.append(record)

        np.savez_compressed(
            output_dir / "sample_{:04d}_layer{}_maps.npz".format(sample_index, args.layer),
            display_image=record["image"],
            student_curvature=record["student_raw"],
            student_importance_normalized=record["student_unit"],
            part_attention=record["part_raw"],
            part_excess_normalized=record["part_unit"],
            part_consensus_normalized=record["consensus_unit"],
            query_agreement=record["agreement"],
            part_overlap=record["overlap"],
        )

        status = record["metrics"]["interpretation_status"]
        overlap = record["metrics"]["mean_pairwise_overlap"]
        print(
            "[sample {:04d}] correct={} confidence={:.4f} overlap={:.6f} status={}".format(
                sample_index,
                record["metrics"]["correct"],
                record["metrics"]["confidence"],
                overlap,
                status,
            )
        )
        if status == "high_query_redundancy":
            print(
                "[warning] Part queries are nearly identical; report shared evidence consensus, "
                "not distinct anatomical parts."
            )
        if args.diagnostics:
            plot_query_diagnostics(
                record,
                output_dir / "sample_{:04d}_layer{}".format(sample_index, args.layer),
                args.formats,
                args.overlay_alpha,
            )

    plot_gallery(
        records,
        output_dir / "visual_analysis_layer{}".format(args.layer),
        args.formats,
        args.overlay_alpha,
        args.layer,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "layer": int(args.layer),
        "sample_indices": [int(value) for value in args.sample_indices],
        "normalization": {
            "student": "positive response above the within-image minimum; 99th-percentile scale",
            "part": "positive attention mass above the uniform 1/N reference; 99th-percentile scale",
            "agreement": "fraction of queries whose location lies in that query's top fraction",
        },
        "paper_policy": {
            "teacher_student_single_image": "excluded; use dataset-level fig:distillation_fidelity",
            "semantic_position_heatmap": "excluded; exact tokenizer labels and padding mask required",
            "high_overlap": "consensus evidence only; never claim distinct parts",
        },
        "samples": [record["metrics"] for record in records],
    }
    with (output_dir / "visual_analysis_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("[done] {}".format(output_dir.resolve()))


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
