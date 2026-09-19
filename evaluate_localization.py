"""Quantitative localization evaluation for Curv-Part on CUB-200-2011.

Metrics:
  * box/mask pointing game
  * top-fraction pixel IoU and predicted-box IoU
  * foreground energy concentration and area-normalized concentration gain
  * top-k foreground precision
  * Hungarian-matched part NME/PCK and GT-part coverage (diagnostic only)
  * unconstrained Softmax evidence-token consensus and border bias
  * optional causal deletion against matched random foreground/background
  * optional horizontal-flip consensus stability

The script uses ``part_attn`` (the pre-dropout attention that actually forms
part tokens), not ``part_assign`` (semantic token-to-part compatibility).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

from localization_metrics import (
    aggregate_evidence_maps,
    decompose_part_attention_logits,
    diagnose_content_attention,
    evaluate_heatmap,
    evaluate_evidence_consensus,
    evaluate_part_points,
    foreground_energy_fraction,
    part_peak_points,
    select_top_evidence_tokens,
    select_evaluation_indices,
    square_deletion_mask,
    validate_evidence_attention,
)
from visualize import build_loader, build_model, load_config


class _Dummy(dict):
    def __getattr__(self, name):
        return _Dummy()

    def __call__(self, *args, **kwargs):
        return _Dummy()


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Dummy


_pm = types.ModuleType("safepickle")
_pm.Unpickler = _SafeUnpickler
_NEAREST = getattr(Image, "Resampling", Image).NEAREST


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", required=True, help="Saved YAML/config.json used for training")
    parser.add_argument("--ckpt", required=True, help="best.pth/latest.pth checkpoint")
    parser.add_argument("--out", default="./localization_eval")
    parser.add_argument("--layer", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--map-sources",
        nargs="+",
        default=["curvature", "curv_weight", "part_attention"],
        choices=("curvature", "curv_weight", "part_attention", "hvp_curvature"),
        help="Heatmaps to evaluate; HVP is optional and substantially slower.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 evaluates the full test set")
    parser.add_argument(
        "--sample-mode",
        choices=("stratified", "random", "sequential"),
        default="stratified",
        help="Sampling used when --max-images is set. Stratified avoids CUB's class-sorted prefix bias.",
    )
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mask-dir",
        default=None,
        help="Optional true foreground masks, mirroring CUB relative paths with .png suffix. "
             "Without this option, foreground metrics use official CUB boxes.",
    )
    parser.add_argument("--save-overlays", type=int, default=20)
    parser.add_argument(
        "--save-attention-maps",
        type=int,
        default=None,
        help="Number of continuous evidence-attention figures to save. "
             "Defaults to --save-overlays for backward compatibility.",
    )
    parser.add_argument(
        "--visualize-top-k",
        type=int,
        default=4,
        help="Number of evidence-token maps shown per image; does not affect quantitative evaluation.",
    )
    parser.add_argument(
        "--visualize-token-selection",
        choices=("fixed", "top_peak"),
        default="fixed",
        help="Token panels to show. 'fixed' uses stable token indices and avoids per-image cherry-picking; "
             "'top_peak' is retained only as an explicitly labelled diagnostic.",
    )
    parser.add_argument(
        "--visualize-token-ids",
        nargs="+",
        type=int,
        default=None,
        help="Exact evidence-token indices to display, e.g. 0 2 5. Overrides --visualize-top-k/selection.",
    )
    parser.add_argument(
        "--attention-aggregation",
        choices=("mean", "max"),
        default="mean",
        help="Aggregation displayed beside the input image. Both mean and max are evaluated quantitatively.",
    )
    parser.add_argument(
        "--attention-interpolation",
        choices=("nearest", "bilinear"),
        default="nearest",
        help="Rendering only. Nearest preserves the original token grid; metrics always use bilinear dense maps.",
    )
    parser.add_argument(
        "--attention-decomposition",
        action="store_true",
        help="Decompose the same Part-attention logits into content, semantic, curvature, and final terms.",
    )
    parser.add_argument(
        "--save-decomposition-maps",
        type=int,
        default=20,
        help="Number of decomposition figures saved when --attention-decomposition is enabled.",
    )
    parser.add_argument(
        "--content-norm-diagnostic",
        action="store_true",
        help="Compare raw q-k dot-product attention with equal-sharpness cosine attention and key norms.",
    )
    parser.add_argument(
        "--save-content-norm-maps",
        type=int,
        default=20,
        help="Number of raw/cosine/key-norm diagnostic figures to save.",
    )
    parser.add_argument(
        "--causal-deletion",
        action="store_true",
        help="Mask the consensus evidence patch and compare confidence drops with random foreground/background patches.",
    )
    parser.add_argument(
        "--deletion-size",
        type=float,
        default=0.15,
        help="Side length of the square deletion patch as a fraction of image size.",
    )
    parser.add_argument("--deletion-random-samples", type=int, default=5)
    parser.add_argument("--deletion-batch-size", type=int, default=4)
    parser.add_argument(
        "--flip-stability",
        action="store_true",
        help="Measure consensus-peak stability under a horizontal flip.",
    )
    return parser.parse_args()


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_single_process_group():
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29513")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_pm)
    except TypeError:  # PyTorch 1.x has no weights_only argument
        return torch.load(path, map_location="cpu", pickle_module=_pm)


def find_aux(output):
    if isinstance(output, dict):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, dict):
                return item
    raise RuntimeError("model output does not contain an auxiliary dictionary")


def find_logits(output):
    if torch.is_tensor(output) and output.ndim == 2:
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 2:
                return item
    raise RuntimeError("model output does not contain [B,C] classification logits")


def random_points_from_mask(mask, count, rng):
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if len(xs) == 0 or count <= 0:
        return []
    ids = rng.integers(0, len(xs), size=count)
    return [(float(xs[i]) + 0.5, float(ys[i]) + 0.5) for i in ids]


def causal_deletion_metrics(
    model,
    image,
    meta_sample,
    original_logits,
    target,
    consensus_point,
    foreground,
    deletion_size,
    random_samples,
    deletion_batch_size,
    rng,
):
    """Compare a consensus-evidence deletion with matched random patches."""
    _, _, height, width = image.shape
    centers = [("consensus", consensus_point)]
    centers.extend(
        ("random_foreground", p)
        for p in random_points_from_mask(foreground, random_samples, rng)
    )
    centers.extend(
        ("random_background", p)
        for p in random_points_from_mask(~np.asarray(foreground, dtype=bool), random_samples, rng)
    )

    variants = image.repeat(len(centers), 1, 1, 1).clone()
    for index, (_kind, point) in enumerate(centers):
        deletion = square_deletion_mask((height, width), point, deletion_size)
        deletion = torch.from_numpy(deletion).to(device=image.device)
        # Inputs are ImageNet-normalized, so zero is the channel-wise mean.
        variants[index, :, deletion] = 0.0

    masked_logits = []
    previous_assess = getattr(model, "assess", None)
    if previous_assess is not None:
        model.assess = False
    try:
        with torch.no_grad():
            for start in range(0, len(variants), max(1, deletion_batch_size)):
                stop = min(start + max(1, deletion_batch_size), len(variants))
                meta_chunk = None
                if meta_sample is not None:
                    repeats = [stop - start] + [1] * (meta_sample.ndim - 1)
                    meta_chunk = meta_sample.repeat(*repeats)
                output = model(variants[start:stop], meta_chunk, return_aux=False)
                masked_logits.append(find_logits(output).detach())
    finally:
        if previous_assess is not None:
            model.assess = previous_assess

    masked_logits = torch.cat(masked_logits, dim=0)
    original_logits = original_logits.detach().float()
    original_prob = torch.softmax(original_logits, dim=-1)
    masked_prob = torch.softmax(masked_logits.float(), dim=-1)
    target = int(target)
    predicted = int(original_logits.argmax())

    result = {
        "causal_original_target_probability": float(original_prob[target]),
        "causal_original_predicted_probability": float(original_prob[predicted]),
        "causal_original_correct": float(predicted == target),
    }
    grouped = defaultdict(list)
    for index, (kind, _point) in enumerate(centers):
        grouped[kind].append(index)

    for kind, ids in grouped.items():
        target_prob = masked_prob[ids, target].mean()
        pred_prob = masked_prob[ids, predicted].mean()
        target_logit = masked_logits[ids, target].mean()
        result[f"causal_{kind}_target_probability_drop"] = float(original_prob[target] - target_prob)
        result[f"causal_{kind}_predicted_probability_drop"] = float(original_prob[predicted] - pred_prob)
        result[f"causal_{kind}_target_logit_drop"] = float(original_logits[target] - target_logit)

    consensus_drop = result["causal_consensus_target_probability_drop"]
    for baseline in ("random_foreground", "random_background"):
        key = f"causal_{baseline}_target_probability_drop"
        if key in result:
            result[f"causal_consensus_minus_{baseline}_target_probability_drop"] = (
                consensus_drop - result[key]
            )
    return result


def find_cub_root(dataset) -> Path:
    root = Path(getattr(dataset, "root", ""))
    candidates = [root, root / "CUB_200_2011"]
    if getattr(dataset, "samples", None):
        sample = Path(dataset.samples[0][0]).resolve()
        candidates.extend(sample.parents)
    for candidate in candidates:
        if (candidate / "images.txt").is_file() and (candidate / "bounding_boxes.txt").is_file():
            return candidate
    raise FileNotFoundError("cannot locate CUB_200_2011 metadata (images.txt/bounding_boxes.txt)")


class CUBAnnotations:
    def __init__(self, cub_root: Path):
        self.root = Path(cub_root)
        self.rel_to_id = {}
        with open(self.root / "images.txt", "r", encoding="utf-8") as handle:
            for line in handle:
                image_id, relative_path = line.strip().split(maxsplit=1)
                self.rel_to_id[relative_path.replace("\\", "/")] = int(image_id)

        self.boxes = {}
        with open(self.root / "bounding_boxes.txt", "r", encoding="utf-8") as handle:
            for line in handle:
                image_id, x, y, width, height = line.strip().split()
                self.boxes[int(image_id)] = tuple(map(float, (x, y, width, height)))

        self.parts = defaultdict(list)
        part_file = self.root / "parts" / "part_locs.txt"
        with open(part_file, "r", encoding="utf-8") as handle:
            for line in handle:
                image_id, part_id, x, y, visible = line.strip().split()
                if int(visible) == 1:
                    self.parts[int(image_id)].append((int(part_id), float(x), float(y)))

    def image_id(self, image_path: str) -> int:
        normalized = str(image_path).replace("\\", "/")
        marker = "/images/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
        else:
            relative = Path(image_path).name
        if relative not in self.rel_to_id:
            raise KeyError(f"image is absent from CUB images.txt: {relative}")
        return self.rel_to_id[relative]


def eval_geometry(width, height, image_size, crop):
    """Return (scale_x, scale_y, offset_x, offset_y) for the eval transform."""
    if crop:
        resize_short = int((256 / 224) * image_size)
        if width <= height:
            resized_w = resize_short
            resized_h = int(resize_short * height / width)
        else:
            resized_h = resize_short
            resized_w = int(resize_short * width / height)
        left = int(round((resized_w - image_size) / 2.0))
        top = int(round((resized_h - image_size) / 2.0))
        return resized_w / width, resized_h / height, -left, -top
    return image_size / width, image_size / height, 0.0, 0.0


def transform_point(x, y, geometry):
    sx, sy, ox, oy = geometry
    return x * sx + ox, y * sy + oy


def make_box_mask(box, original_size, image_size, crop):
    width, height = original_size
    x, y, box_w, box_h = box
    geometry = eval_geometry(width, height, image_size, crop)
    x1, y1 = transform_point(x, y, geometry)
    x2, y2 = transform_point(x + box_w, y + box_h, geometry)
    x1, x2 = np.clip([x1, x2], 0, image_size)
    y1, y2 = np.clip([y1, y2], 0, image_size)
    mask = np.zeros((image_size, image_size), dtype=bool)
    ix1, iy1 = int(math.floor(x1)), int(math.floor(y1))
    ix2, iy2 = int(math.ceil(x2)), int(math.ceil(y2))
    mask[iy1:iy2, ix1:ix2] = True
    return mask, geometry, math.hypot(max(x2 - x1, 1.0), max(y2 - y1, 1.0))


def transform_visible_parts(parts, geometry, image_size):
    output = []
    for _part_id, x, y in parts:
        tx, ty = transform_point(x, y, geometry)
        if 0.0 <= tx < image_size and 0.0 <= ty < image_size:
            output.append((tx, ty))
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def load_custom_mask(mask_dir, relative_path, image_size, crop):
    if not mask_dir:
        return None
    relative = Path(relative_path).with_suffix(".png")
    path = Path(mask_dir) / relative
    if not path.is_file():
        return None
    mask = Image.open(path).convert("L")
    if crop:
        resize_short = int((256 / 224) * image_size)
        w, h = mask.size
        if w <= h:
            new_size = (resize_short, int(resize_short * h / w))
        else:
            new_size = (int(resize_short * w / h), resize_short)
        mask = mask.resize(new_size, resample=_NEAREST)
        left = int(round((mask.width - image_size) / 2.0))
        top = int(round((mask.height - image_size) / 2.0))
        mask = mask.crop((left, top, left + image_size, top + image_size))
    else:
        mask = mask.resize((image_size, image_size), resample=_NEAREST)
    return np.asarray(mask) > 0


def tensor_to_token_vector(x, sample_index):
    value = x[sample_index].detach().float().cpu().squeeze()
    if value.ndim == 1:
        return value.numpy()
    if value.ndim == 2 and 1 in value.shape:
        return value.reshape(-1).numpy()
    raise ValueError(f"cannot interpret token map shape={tuple(value.shape)}")


def token_vector_to_dense(vector, output_size):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    side = int(round(math.sqrt(len(vector))))
    if side * side != len(vector):
        raise ValueError(f"token count {len(vector)} is not square")
    tensor = torch.from_numpy(vector.reshape(1, 1, side, side))
    return F.interpolate(tensor, size=(output_size, output_size), mode="bilinear", align_corners=False)[0, 0].numpy()


def part_tensor_to_maps(part_tensor, sample_index):
    maps = part_tensor[sample_index].detach().float().cpu().numpy()
    if maps.ndim != 2:
        raise ValueError(f"part attention must be [P,N] or [N,P], got {maps.shape}")
    # Actual part_attn is [P,N]. Retain a defensive transpose for older checkpoints.
    if maps.shape[0] > maps.shape[1] and int(round(math.sqrt(maps.shape[0]))) ** 2 == maps.shape[0]:
        maps = maps.T
    side = int(round(math.sqrt(maps.shape[1])))
    if side * side != maps.shape[1]:
        raise ValueError(f"part attention token count {maps.shape[1]} is not square")
    return maps.reshape(maps.shape[0], side, side)


def relative_cub_path(image_path):
    normalized = str(image_path).replace("\\", "/")
    return normalized.split("/images/", 1)[1] if "/images/" in normalized else Path(image_path).name


def input_tensor_to_pil(image_tensor):
    image = image_tensor.detach().float().cpu().clone()
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    image = (image * std + mean).clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _viridis_rgb(heatmap, vmin=None, vmax=None):
    """Small dependency-free approximation of the perceptually uniform viridis map."""
    h = np.asarray(heatmap, dtype=np.float64)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(h.min()) if vmin is None else float(vmin)
    hi = float(h.max()) if vmax is None else float(vmax)
    h = np.clip((h - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    anchors = np.asarray([
        [68, 1, 84], [59, 82, 139], [33, 145, 140],
        [94, 201, 98], [253, 231, 37],
    ], dtype=np.float64)
    position = h * (len(anchors) - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    weight = (position - lower)[..., None]
    return np.round(anchors[lower] * (1.0 - weight) + anchors[upper] * weight).astype(np.uint8)


def _load_figure_font(size=18):
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def save_attention_figure(
    path,
    evaluation_image,
    part_maps,
    top_k=4,
    aggregation="mean",
    interpolation="nearest",
    token_selection="fixed",
    token_ids=None,
):
    """Save input + aggregate + automatically selected evidence-token maps.

    Individual panels are normalized only for rendering. Raw pre-dropout
    Softmax attention remains untouched for every quantitative metric.
    """
    maps = np.asarray(part_maps, dtype=np.float64)
    if token_ids is not None:
        selected = np.asarray(token_ids, dtype=np.int64)
        if len(selected) and (selected.min() < 0 or selected.max() >= len(maps)):
            raise ValueError(
                f"visualized token ids must be in [0,{len(maps) - 1}], got {selected.tolist()}"
            )
    elif token_selection == "fixed":
        selected = np.arange(min(int(top_k), len(maps)), dtype=np.int64)
    elif token_selection == "top_peak":
        selected = select_top_evidence_tokens(maps, top_k)
    else:
        raise ValueError("token_selection must be 'fixed' or 'top_peak'")
    aggregate = aggregate_evidence_maps(maps, aggregation)
    panels = [("Input", evaluation_image.convert("RGB"))]
    panel_size = 256
    resampling = getattr(Image, "Resampling", Image)
    render_resample = resampling.NEAREST if interpolation == "nearest" else resampling.BILINEAR

    shown_maps = [aggregate] + [maps[token_id] for token_id in selected]
    # All part-attention maps are spatial probability distributions, so one
    # shared zero-based scale is meaningful and prevents per-panel contrast
    # stretching from exaggerating tiny differences.
    shared_vmax = max(float(np.nanmax(value)) for value in shown_maps)

    def render_map(value):
        colored = Image.fromarray(_viridis_rgb(value, vmin=0.0, vmax=shared_vmax), mode="RGB")
        return colored.resize((panel_size, panel_size), resample=render_resample)

    panels.append((f"{aggregation.title()} Evidence", render_map(aggregate)))
    for token_id in selected:
        peak = float(np.nanmax(maps[token_id]))
        panels.append((f"Token {int(token_id)}  max={peak:.3g}", render_map(maps[token_id])))

    title_height = 36
    gap = 8
    canvas = Image.new(
        "RGB",
        (len(panels) * panel_size + (len(panels) - 1) * gap, panel_size + title_height),
        "white",
    )
    font = _load_figure_font(17)
    draw = ImageDraw.Draw(canvas)
    for panel_index, (title, panel) in enumerate(panels):
        x = panel_index * (panel_size + gap)
        if title == "Input":
            panel = panel.resize((panel_size, panel_size), resample=resampling.BILINEAR)
        canvas.paste(panel, (x, title_height))
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
        else:
            text_width = draw.textsize(title, font=font)[0]
        draw.text((x + max(0, (panel_size - text_width) // 2), 8), title, fill="black", font=font)
    canvas.save(path)


def save_decomposition_figure(path, evaluation_image, component_maps, interpolation="nearest"):
    """Save probability maps induced by each additive attention-logit term."""
    order = ("content", "semantic", "curvature", "final")
    maps = {name: np.asarray(component_maps[name], dtype=np.float64).mean(axis=0) for name in order}
    panel_size = 256
    title_height = 36
    gap = 8
    resampling = getattr(Image, "Resampling", Image)
    render_resample = resampling.NEAREST if interpolation == "nearest" else resampling.BILINEAR
    shared_vmax = max(float(np.nanmax(value)) for value in maps.values())
    panels = [("Input", evaluation_image.convert("RGB"))]
    for name in order:
        value = maps[name]
        colored = Image.fromarray(
            _viridis_rgb(value, vmin=0.0, vmax=shared_vmax), mode="RGB"
        ).resize((panel_size, panel_size), resample=render_resample)
        panels.append((f"{name.title()}  max={float(value.max()):.3g}", colored))

    canvas = Image.new(
        "RGB",
        (len(panels) * panel_size + (len(panels) - 1) * gap, panel_size + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _load_figure_font(17)
    for panel_index, (title, panel) in enumerate(panels):
        x = panel_index * (panel_size + gap)
        if title == "Input":
            panel = panel.resize((panel_size, panel_size), resample=resampling.BILINEAR)
        canvas.paste(panel, (x, title_height))
        bbox = draw.textbbox((0, 0), title, font=font) if hasattr(draw, "textbbox") else None
        text_width = bbox[2] - bbox[0] if bbox else draw.textsize(title, font=font)[0]
        draw.text((x + max(0, (panel_size - text_width) // 2), 8), title, fill="black", font=font)
    canvas.save(path)


def save_content_norm_figure(path, evaluation_image, diagnostic_maps, final_maps, interpolation="nearest"):
    """Save raw content, matched cosine content, key norm, and final attention."""
    named_maps = {
        "Raw Content": np.asarray(diagnostic_maps["raw_content"]).mean(axis=0),
        "Cosine Content": np.asarray(diagnostic_maps["cosine_content"]).mean(axis=0),
        "Key Norm": np.asarray(diagnostic_maps["key_norm"]).mean(axis=0),
        "Final": np.asarray(final_maps).mean(axis=0),
    }
    panel_size, title_height, gap = 256, 36, 8
    resampling = getattr(Image, "Resampling", Image)
    render_resample = resampling.NEAREST if interpolation == "nearest" else resampling.BILINEAR
    shared_vmax = max(float(np.nanmax(value)) for value in named_maps.values())
    panels = [("Input", evaluation_image.convert("RGB"))]
    for name, value in named_maps.items():
        panel = Image.fromarray(
            _viridis_rgb(value, vmin=0.0, vmax=shared_vmax), mode="RGB"
        ).resize((panel_size, panel_size), resample=render_resample)
        panels.append((f"{name}  max={float(value.max()):.3g}", panel))
    canvas = Image.new(
        "RGB",
        (len(panels) * panel_size + (len(panels) - 1) * gap, panel_size + title_height),
        "white",
    )
    draw, font = ImageDraw.Draw(canvas), _load_figure_font(17)
    for index, (title, panel) in enumerate(panels):
        x = index * (panel_size + gap)
        if title == "Input":
            panel = panel.resize((panel_size, panel_size), resample=resampling.BILINEAR)
        canvas.paste(panel, (x, title_height))
        bbox = draw.textbbox((0, 0), title, font=font) if hasattr(draw, "textbbox") else None
        width = bbox[2] - bbox[0] if bbox else draw.textsize(title, font=font)[0]
        draw.text((x + max(0, (panel_size - width) // 2), 8), title, fill="black", font=font)
    canvas.save(path)


def summarize(rows, bootstrap_samples, seed):
    rng = np.random.default_rng(seed)
    metadata_keys = {"image_id", "target", "layer", "visible_parts"}
    numeric_keys = sorted({
        key for row in rows for key, value in row.items()
        if key not in metadata_keys and isinstance(value, (int, float))
    })
    summary = []
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows if key in row and np.isfinite(row[key])], dtype=np.float64)
        if len(values) == 0:
            continue
        if bootstrap_samples > 0:
            sampled = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
            ci_low, ci_high = np.percentile(sampled, [2.5, 97.5])
        else:
            ci_low = ci_high = float("nan")
        summary.append({
            "metric": key,
            "n": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
        })
    return summary


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0,1]")
    if not 0.0 < args.deletion_size <= 1.0:
        raise ValueError("--deletion-size must be in (0,1]")
    if args.deletion_random_samples < 1:
        raise ValueError("--deletion-random-samples must be at least 1")
    if args.deletion_batch_size < 1:
        raise ValueError("--deletion-batch-size must be at least 1")
    if args.visualize_top_k < 0:
        raise ValueError("--visualize-top-k must be non-negative")
    if args.visualize_token_ids is not None and len(set(args.visualize_token_ids)) != len(args.visualize_token_ids):
        raise ValueError("--visualize-token-ids must not contain duplicates")
    set_deterministic(args.seed)
    init_single_process_group()
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.cfg)
    cfg.defrost()
    cfg.EVAL_MODE = True
    cfg.MODEL.assess = True
    cfg.DATA.BATCH_SIZE = args.batch_size
    cfg.DATA.NUM_WORKERS = args.num_workers
    cfg.freeze()

    checkpoint = load_checkpoint(args.ckpt)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    if "head.weight" in state_dict:
        cfg.defrost()
        cfg.MODEL.NUM_CLASSES = int(state_dict["head.weight"].shape[0])
        cfg.freeze()

    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[checkpoint] missing={len(missing)}, unexpected={len(unexpected)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    model.assess = True

    _, dataset_val, _, _, _ = build_loader(cfg)
    if getattr(dataset_val, "dataset", None) is not None and not hasattr(dataset_val, "samples"):
        dataset_val = dataset_val.dataset
    if cfg.DATA.DATASET != "cub-200":
        raise ValueError("official box/part evaluation currently requires --dataset cub-200 in the config")

    # Do not reuse build_loader's SubsetRandomSampler: sequential order is
    # required to keep sample paths and annotations exactly aligned.
    labels = [int(sample[1]) for sample in dataset_val.samples]
    selected_indices = select_evaluation_indices(
        labels, args.max_images, mode=args.sample_mode, seed=args.seed
    )
    evaluation_dataset = (
        dataset_val
        if len(selected_indices) == len(dataset_val)
        else Subset(dataset_val, selected_indices.tolist())
    )
    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(getattr(cfg.DATA, "PIN_MEMORY", True)),
        drop_last=False,
    )
    annotations = CUBAnnotations(find_cub_root(dataset_val))
    image_size = int(cfg.DATA.IMG_SIZE)
    crop = bool(cfg.TEST.CROP)
    samples_meta = [dataset_val.samples[int(index)] for index in selected_indices]
    use_hvp = "hvp_curvature" in args.map_sources

    rows = []
    global_index = 0
    for batch_index, batch in enumerate(loader):
        if args.max_images and global_index >= args.max_images:
            break
        if cfg.DATA.ADD_META:
            images, targets, meta = batch
            meta = torch.stack([item.float() for item in meta], dim=0).to(device)
        else:
            images, targets = batch
            meta = None
        images = images.to(device, non_blocking=True)

        grad_context = torch.enable_grad() if use_hvp else torch.no_grad()
        with grad_context:
            output = model(
                images,
                meta,
                return_aux=True,
                force_hvp=use_hvp,
                force_hvp_layer=args.layer if use_hvp else None,
            )
        logits = find_logits(output)
        aux = find_aux(output)
        part_attention = aux.get(f"part_attn_{args.layer}")
        if part_attention is None:
            raise KeyError(f"missing part_attn_{args.layer}; update MetaFG_meta.py to expose pre-dropout part attention")

        decomposition_inputs = None
        if args.attention_decomposition:
            required = {
                "attn_logits": aux.get(f"attn_logits_{args.layer}"),
                "token_part_sim": aux.get(f"token_part_sim_{args.layer}"),
                "curvature": aux.get(f"curvature_{args.layer}"),
                "gate_status": aux.get(f"gate_status_{args.layer}"),
            }
            missing_decomposition = [key for key, value in required.items() if value is None]
            if missing_decomposition:
                raise KeyError(
                    "attention decomposition requires missing aux values: "
                    + ", ".join(missing_decomposition)
                )
            decomposition_inputs = required

        content_norm_inputs = None
        if args.content_norm_diagnostic:
            required = {
                "content_logits": aux.get(f"content_logits_{args.layer}"),
                "content_cosine_logits": aux.get(f"content_cosine_logits_{args.layer}"),
                "key_norm": aux.get(f"key_norm_{args.layer}"),
            }
            missing_content_norm = [key for key, value in required.items() if value is None]
            if missing_content_norm:
                raise KeyError(
                    "content norm diagnostic requires the updated model aux values: "
                    + ", ".join(missing_content_norm)
                )
            content_norm_inputs = required

        flipped_part_attention = None
        if args.flip_stability:
            with torch.no_grad():
                flipped_output = model(
                    torch.flip(images, dims=[-1]),
                    meta,
                    return_aux=True,
                    force_hvp=False,
                    force_hvp_layer=None,
                )
            flipped_aux = find_aux(flipped_output)
            flipped_part_attention = flipped_aux.get(f"part_attn_{args.layer}")
            if flipped_part_attention is None:
                raise KeyError(f"missing flipped part_attn_{args.layer}")

        for local_index in range(images.shape[0]):
            if args.max_images and global_index >= args.max_images:
                break
            image_path = samples_meta[global_index][0]
            image_id = annotations.image_id(image_path)
            relative_path = relative_cub_path(image_path)
            with Image.open(image_path) as original:
                original_size = original.size
            box_mask, geometry, box_diagonal = make_box_mask(
                annotations.boxes[image_id], original_size, image_size, crop
            )
            gt_parts = transform_visible_parts(annotations.parts[image_id], geometry, image_size)
            custom_mask = load_custom_mask(args.mask_dir, relative_path, image_size, crop)
            if args.mask_dir and custom_mask is None:
                expected = Path(args.mask_dir) / Path(relative_path).with_suffix(".png")
                raise FileNotFoundError(f"missing requested foreground mask: {expected}")
            foreground = custom_mask if custom_mask is not None else box_mask
            foreground_kind = "segmentation_mask" if custom_mask is not None else "bounding_box"

            part_maps = part_tensor_to_maps(part_attention, local_index)
            pred_parts = part_peak_points(part_maps, (image_size, image_size))
            evidence = evaluate_evidence_consensus(
                part_maps,
                foreground,
                gt_points=gt_parts,
                normalization_length=box_diagonal,
            )
            consensus_point = (evidence["consensus_x"], evidence["consensus_y"])
            row = {
                "image_id": image_id,
                "image": relative_path,
                "target": int(targets[local_index]),
                "layer": args.layer,
                "foreground_kind": foreground_kind,
                "visible_parts": int(len(gt_parts)),
                "classification_top1": float(int(logits[local_index].argmax()) == int(targets[local_index])),
                "classification_top5": float(
                    int(targets[local_index]) in logits[local_index].topk(
                        min(5, logits.shape[-1])
                    ).indices.tolist()
                ),
            }
            if args.visualize_token_ids is not None:
                selected_for_visualization = np.asarray(args.visualize_token_ids, dtype=np.int64)
            elif args.visualize_token_selection == "fixed":
                selected_for_visualization = np.arange(
                    min(args.visualize_top_k, len(part_maps)), dtype=np.int64
                )
            else:
                selected_for_visualization = select_top_evidence_tokens(
                    part_maps, args.visualize_top_k
                )
            row["visualized_token_ids"] = ";".join(
                str(int(token_id)) for token_id in selected_for_visualization
            )
            attention_check = validate_evidence_attention(part_maps)
            row.update({
                f"part_attention_check_{key}": value
                for key, value in attention_check.items()
            })
            row.update({f"evidence_{key}": value for key, value in evidence.items()})

            component_maps = None
            if decomposition_inputs is not None:
                gates = decomposition_inputs["gate_status"]
                component_maps, component_diagnostics = decompose_part_attention_logits(
                    decomposition_inputs["attn_logits"][local_index].detach().float().cpu().numpy(),
                    decomposition_inputs["token_part_sim"][local_index].detach().float().cpu().numpy(),
                    decomposition_inputs["curvature"][local_index].detach().float().cpu().numpy(),
                    similarity_gate=float(gates["sim_logit_gate"]),
                    curvature_gate=float(gates["curv_logit_gate"]),
                    observed_attention=part_maps.reshape(len(part_maps), -1),
                )
                row.update({
                    f"decomposition_{key}": value
                    for key, value in component_diagnostics.items()
                })
                for component_name, component_part_maps in component_maps.items():
                    component_grid = component_part_maps.reshape(part_maps.shape)
                    component_mean = aggregate_evidence_maps(component_grid, "mean")
                    dense_component = token_vector_to_dense(component_mean.reshape(-1), image_size)
                    component_metrics = evaluate_heatmap(
                        dense_component, foreground, args.top_fraction
                    )
                    row.update({
                        f"decomposition_{component_name}_{key}": value
                        for key, value in component_metrics.items()
                    })

            content_norm_maps = None
            if content_norm_inputs is not None:
                content_norm_maps, content_norm_diagnostics = diagnose_content_attention(
                    content_norm_inputs["content_logits"][local_index].detach().float().cpu().numpy(),
                    content_norm_inputs["content_cosine_logits"][local_index].detach().float().cpu().numpy(),
                    content_norm_inputs["key_norm"][local_index].detach().float().cpu().numpy(),
                )
                row.update({
                    f"content_norm_{key}": value
                    for key, value in content_norm_diagnostics.items()
                })
                for diagnostic_name, diagnostic_part_maps in content_norm_maps.items():
                    grid = diagnostic_part_maps.reshape(
                        diagnostic_part_maps.shape[0], part_maps.shape[1], part_maps.shape[2]
                    )
                    mean_map = aggregate_evidence_maps(grid, "mean")
                    dense_map = token_vector_to_dense(mean_map.reshape(-1), image_size)
                    diagnostic_metrics = evaluate_heatmap(dense_map, foreground, args.top_fraction)
                    row.update({
                        f"content_norm_{diagnostic_name}_{key}": value
                        for key, value in diagnostic_metrics.items()
                    })

            if flipped_part_attention is not None:
                flipped_maps = part_tensor_to_maps(flipped_part_attention, local_index)
                flipped_evidence = evaluate_evidence_consensus(
                    flipped_maps,
                    np.fliplr(foreground),
                )
                reflected_x = image_size - flipped_evidence["consensus_x"]
                reflected_y = flipped_evidence["consensus_y"]
                distance = math.hypot(
                    consensus_point[0] - reflected_x,
                    consensus_point[1] - reflected_y,
                )
                row["evidence_flip_stability_nme"] = distance / (math.sqrt(2.0) * image_size)
                row["evidence_flip_consensus_ratio"] = flipped_evidence["consensus_ratio"]

            source_maps = {}
            for source in args.map_sources:
                if source == "part_attention":
                    token_map = aggregate_evidence_maps(part_maps, args.attention_aggregation)
                    dense = token_vector_to_dense(token_map.reshape(-1), image_size)
                else:
                    tensor = aux.get(f"{source}_{args.layer}")
                    if tensor is None:
                        raise KeyError(f"requested source {source}_{args.layer} is unavailable")
                    dense = token_vector_to_dense(tensor_to_token_vector(tensor, local_index), image_size)
                source_maps[source] = dense
                metrics = evaluate_heatmap(dense, foreground, args.top_fraction)
                row.update({f"{source}_{key}": value for key, value in metrics.items()})
                # Always retain a box-based concentration number, even when an
                # optional segmentation mask is used for the main metrics.
                row[f"{source}_box_foreground_energy"] = foreground_energy_fraction(dense, box_mask)

            part_metrics = evaluate_part_points(pred_parts, gt_parts, box_diagonal, thresholds=(0.10, 0.20))
            row.update({f"part_{key}": value for key, value in part_metrics.items()})

            per_part_energy = []
            per_part_metrics = defaultdict(list)
            for part_map in part_maps:
                dense_part = token_vector_to_dense(part_map.reshape(-1), image_size)
                per_part_energy.append(foreground_energy_fraction(dense_part, foreground))
                for metric_name, metric_value in evaluate_heatmap(
                    dense_part, foreground, args.top_fraction
                ).items():
                    per_part_metrics[metric_name].append(metric_value)
            row["part_foreground_energy_mean"] = float(np.nanmean(per_part_energy))
            row["part_foreground_energy_min"] = float(np.nanmin(per_part_energy))
            for metric_name, values in per_part_metrics.items():
                row[f"part_all_tokens_{metric_name}_mean"] = float(np.nanmean(values))

            # Report both aggregate definitions regardless of which one is
            # selected for the visualization and backward-compatible key.
            for aggregation in ("mean", "max"):
                aggregate_map = aggregate_evidence_maps(part_maps, aggregation)
                dense_aggregate = token_vector_to_dense(aggregate_map.reshape(-1), image_size)
                aggregate_metrics = evaluate_heatmap(dense_aggregate, foreground, args.top_fraction)
                row.update({
                    f"part_attention_{aggregation}_{key}": value
                    for key, value in aggregate_metrics.items()
                })

            if args.causal_deletion:
                meta_sample = None if meta is None else meta[local_index:local_index + 1]
                rng = np.random.default_rng(args.seed + 1009 * image_id + 17 * args.layer)
                row.update(causal_deletion_metrics(
                    model=model,
                    image=images[local_index:local_index + 1],
                    meta_sample=meta_sample,
                    original_logits=logits[local_index],
                    target=int(targets[local_index]),
                    consensus_point=consensus_point,
                    foreground=foreground,
                    deletion_size=args.deletion_size,
                    random_samples=args.deletion_random_samples,
                    deletion_batch_size=args.deletion_batch_size,
                    rng=rng,
                ))
            rows.append(row)

            attention_figure_count = (
                args.save_overlays if args.save_attention_maps is None else args.save_attention_maps
            )
            if global_index < attention_figure_count:
                save_attention_figure(
                    output_dir / f"attention_{global_index:05d}.png",
                    input_tensor_to_pil(images[local_index]),
                    part_maps,
                    top_k=args.visualize_top_k,
                    aggregation=args.attention_aggregation,
                    interpolation=args.attention_interpolation,
                    token_selection=args.visualize_token_selection,
                    token_ids=args.visualize_token_ids,
                )
            if (
                args.attention_decomposition
                and component_maps is not None
                and global_index < args.save_decomposition_maps
            ):
                save_decomposition_figure(
                    output_dir / f"decomposition_{global_index:05d}.png",
                    input_tensor_to_pil(images[local_index]),
                    {
                        name: value.reshape(part_maps.shape)
                        for name, value in component_maps.items()
                    },
                    interpolation=args.attention_interpolation,
                )
            if (
                args.content_norm_diagnostic
                and content_norm_maps is not None
                and global_index < args.save_content_norm_maps
            ):
                save_content_norm_figure(
                    output_dir / f"content_norm_{global_index:05d}.png",
                    input_tensor_to_pil(images[local_index]),
                    {
                        name: value.reshape(value.shape[0], part_maps.shape[1], part_maps.shape[2])
                        for name, value in content_norm_maps.items()
                    },
                    part_maps,
                    interpolation=args.attention_interpolation,
                )
            global_index += 1

        if batch_index % 20 == 0:
            print(f"[progress] evaluated {global_index} images")

    summary = summarize(rows, args.bootstrap_samples, args.seed)
    write_csv(output_dir / "localization_per_image.csv", rows)
    write_csv(output_dir / "localization_summary.csv", summary)
    with open(output_dir / "localization_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": vars(args),
                "num_images": len(rows),
                "foreground_default": "custom segmentation mask when available, otherwise CUB bounding box",
                "iou_definition": f"exact top-{args.top_fraction:.0%} pixels; pred_box_iou is the tight box around them",
                "part_attention_definition": "pre-dropout attention used to form part tokens",
                "attention_visualization_definition": (
                    "input, selected aggregate, and a compact subset of evidence-token Softmax maps; "
                    "fixed token ids and a shared zero-based color scale are the default, while all tokens are evaluated"
                ),
                "attention_decomposition_definition": (
                    "final Part-attention logits reconstructed as content + gated semantic similarity + "
                    "gated log1p curvature; component maps are spatial Softmax diagnostics of the same logits"
                ),
                "content_norm_diagnostic_definition": (
                    "raw q-k content attention versus cosine q-k attention rescaled to equal centered RMS, "
                    "plus the spatial key-norm distribution; diagnostics do not change model predictions"
                ),
                "evidence_consensus_definition": (
                    "modal spatial peak across unconstrained Softmax evidence tokens; "
                    "agreement is descriptive and is not treated as a diversity objective"
                ),
                "causal_deletion_definition": (
                    "target/predicted-class confidence drop after masking a fixed-area consensus patch; "
                    "positive consensus-minus-random values support causal discriminativeness"
                ),
                "flip_stability_definition": "distance between original and inverse-flipped consensus peaks, normalized by image diagonal",
                "metrics": summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\nLocalization evaluation finished")
    print(f"  images:  {len(rows)}")
    print(f"  details: {output_dir / 'localization_per_image.csv'}")
    print(f"  summary: {output_dir / 'localization_summary.csv'}")
    print(f"  json:    {output_dir / 'localization_summary.json'}")
    if rows:
        print(f"  top1:    {100.0 * np.mean([row['classification_top1'] for row in rows]):.3f}%")
        print(f"  top5:    {100.0 * np.mean([row['classification_top5'] for row in rows]):.3f}%")


if __name__ == "__main__":
    main()
