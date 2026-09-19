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

    entropy = -(flat * np.log(np.clip(flat, 1e-12, None))).sum(axis=1)
    normalized_entropy = entropy / max(np.log(float(flat.shape[1])), 1e-12)
    total_variation_from_uniform = 0.5 * np.abs(flat - uniform).sum(axis=1)
    peak_lift_over_uniform = flat.max(axis=1) / max(uniform, 1e-12)

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
        "mean_normalized_attention_entropy": float(normalized_entropy.mean()),
        "min_normalized_attention_entropy": float(normalized_entropy.min()),
        "max_normalized_attention_entropy": float(normalized_entropy.max()),
        "mean_total_variation_from_uniform": float(total_variation_from_uniform.mean()),
        "mean_peak_lift_over_uniform": float(peak_lift_over_uniform.mean()),
        "max_peak_lift_over_uniform": float(peak_lift_over_uniform.max()),
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


def _average_ranks(values):
    """Average ranks with exact ties preserved; no SciPy dependency."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def tied_spearman(left, right):
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left_rank, right_rank) / denominator)


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
    statistics["student_part_consensus_spearman"] = tied_spearman(
        student_grid, consensus
    )
    statistics.update(metadata)
    warnings = []
    if not statistics.get("correct", True):
        warnings.append("misclassified_sample")
    if statistics["mean_normalized_attention_entropy"] >= 0.98:
        warnings.append("diffuse_part_attention")
    correlation = statistics["student_part_consensus_spearman"]
    if np.isfinite(correlation) and correlation <= 0.10:
        warnings.append("weak_student_part_spatial_agreement")
    if statistics["mean_pairwise_overlap"] >= 0.98:
        warnings.append("query_redundancy_distinct_parts_not_supported")
    statistics["paper_warnings"] = warnings
    statistics["positive_main_text_ready"] = bool(
        statistics.get("correct", True)
        and "diffuse_part_attention" not in warnings
        and "weak_student_part_spatial_agreement" not in warnings
    )
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


def _plot_gallery(records, output_stem, formats, alpha, panel_keys, titles, top_fraction):
    rows = len(records)
    columns = len(panel_keys)
    figure_width = 4.8 if columns == 2 else 7.16
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(figure_width, max(1.72 * rows, 1.9)),
        constrained_layout=True,
        squeeze=False,
    )
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="semibold", fontsize=8.2)

    for row, record in enumerate(records):
        image = record["image"]
        panel_lookup = {
            "input": image,
            "student": overlay_evidence(
                image, record["student_unit"], CMAP_EVIDENCE, alpha
            ),
            "part": overlay_evidence(
                image, record["consensus_unit"], CMAP_EVIDENCE, alpha
            ),
            "agreement": overlay_evidence(
                image, record["agreement"], CMAP_AGREEMENT, alpha
            ),
        }
        panels = [panel_lookup[key] for key in panel_keys]
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

    evidence_columns = [
        index for index, key in enumerate(panel_keys) if key in {"student", "part"}
    ]
    if evidence_columns:
        evidence_bar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP_EVIDENCE)
        cb1 = fig.colorbar(
            evidence_bar,
            ax=axes[:, evidence_columns].ravel().tolist(),
            orientation="horizontal",
            fraction=0.025,
            pad=0.015,
            aspect=50,
        )
        cb1.set_label("Within-image normalized evidence", labelpad=2)
    agreement_columns = [
        index for index, key in enumerate(panel_keys) if key == "agreement"
    ]
    if agreement_columns:
        agreement_bar = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP_AGREEMENT)
        cb2 = fig.colorbar(
            agreement_bar,
            ax=axes[:, agreement_columns].ravel().tolist(),
            orientation="horizontal",
            fraction=0.025,
            pad=0.015,
            aspect=25,
        )
        cb2.set_label(
            "Query consensus fraction (top {:.0f}%)".format(
                100.0 * float(top_fraction)
            ),
            labelpad=2,
        )
    save_figure(fig, output_stem, formats)


def plot_main_gallery(records, output_stem, formats, alpha, top_fraction):
    """Compact main-text candidate without the diagnostic agreement column."""
    _plot_gallery(
        records,
        output_stem,
        formats,
        alpha,
        ("input", "student", "part"),
        ("(a) Input", "(b) Student importance", "(c) Shared part evidence"),
        top_fraction,
    )


def plot_gallery(records, output_stem, formats, alpha, layer, top_fraction=0.10):
    """Full four-column analysis, retained for backward compatibility."""
    agreement_title = "(d) Query consensus\n(top {:.0f}%)".format(
        100.0 * float(top_fraction)
    )
    _plot_gallery(
        records,
        output_stem,
        formats,
        alpha,
        ("input", "student", "part", "agreement"),
        (
            "(a) Input",
            "(b) Student importance",
            "(c) Shared part evidence",
            agreement_title,
        ),
        top_fraction,
    )


def plot_query_consensus_gallery(records, output_stem, formats, alpha, top_fraction):
    """Large two-column comparison for judging query-consensus readability."""
    agreement_title = "(b) Query consensus (top {:.0f}%)".format(
        100.0 * float(top_fraction)
    )
    _plot_gallery(
        records,
        output_stem,
        formats,
        alpha,
        ("input", "agreement"),
        ("(a) Input", agreement_title),
        top_fraction,
    )


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
            "[sample {:04d}] correct={} confidence={:.4f} overlap={:.6f} "
            "entropy={:.6f} student-part-rho={:.4f} status={}".format(
                sample_index,
                record["metrics"]["correct"],
                record["metrics"]["confidence"],
                overlap,
                record["metrics"]["mean_normalized_attention_entropy"],
                record["metrics"]["student_part_consensus_spearman"],
                status,
            )
        )
        if status == "high_query_redundancy":
            print(
                "[warning] Part queries are nearly identical; report shared evidence consensus, "
                "not distinct anatomical parts."
            )
        for warning in record["metrics"]["paper_warnings"]:
            print("[paper-warning] sample {:04d}: {}".format(sample_index, warning))
        if args.diagnostics:
            plot_query_diagnostics(
                record,
                output_dir / "sample_{:04d}_layer{}".format(sample_index, args.layer),
                args.formats,
                args.overlay_alpha,
            )

    # Produce both choices requested for paper review.  The compact version is
    # the default main-text candidate; the full version preserves the query
    # consensus diagnostic; the two-column version makes that diagnostic easy
    # to judge without shrinking it into a four-column grid.
    plot_main_gallery(
        records,
        output_dir / "visual_analysis_main_layer{}".format(args.layer),
        args.formats,
        args.overlay_alpha,
        args.top_fraction,
    )
    plot_gallery(
        records,
        output_dir / "visual_analysis_with_query_consensus_layer{}".format(args.layer),
        args.formats,
        args.overlay_alpha,
        args.layer,
        args.top_fraction,
    )
    plot_query_consensus_gallery(
        records,
        output_dir / "query_consensus_layer{}".format(args.layer),
        args.formats,
        args.overlay_alpha,
        args.top_fraction,
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "layer": int(args.layer),
        "sample_indices": [int(value) for value in args.sample_indices],
        "normalization": {
            "student": "positive response above the within-image minimum; 99th-percentile scale",
            "part": "positive attention mass above the uniform 1/N reference; 99th-percentile scale",
            "agreement": "fraction of queries whose location lies in that query's top fraction; spatial consensus only, not evidence magnitude",
        },
        "paper_policy": {
            "teacher_student_single_image": "excluded; use dataset-level fig:distillation_fidelity",
            "semantic_position_heatmap": "excluded; exact tokenizer labels and padding mask required",
            "high_overlap": "consensus evidence only; never claim distinct parts",
            "query_consensus": "interpret jointly with shared part evidence; top-k membership alone does not encode attention magnitude",
            "positive_examples": "do not use misclassified or quality-flagged samples as positive main-text evidence",
        },
        "outputs": {
            "main_text_candidate": "visual_analysis_main_layer{}".format(args.layer),
            "full_diagnostic": "visual_analysis_with_query_consensus_layer{}".format(args.layer),
            "query_consensus_standalone": "query_consensus_layer{}".format(args.layer),
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
