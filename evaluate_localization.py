"""Quantitative localization evaluation for Curv-Part on CUB-200-2011.

Metrics:
  * box/mask pointing game
  * top-fraction pixel IoU and predicted-box IoU
  * foreground energy concentration and area-normalized concentration gain
  * top-k foreground precision
  * Hungarian-matched part NME/PCK, GT-part coverage, predicted-part precision,
    and part-peak diversity using CUB's visible part keypoints

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
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from localization_metrics import (
    evaluate_heatmap,
    evaluate_part_points,
    foreground_energy_fraction,
    part_peak_points,
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


def save_overlay(path, evaluation_image, heatmap, box_mask, gt_parts, pred_parts):
    image = evaluation_image.convert("RGB")
    h = heatmap.astype(np.float64)
    h = (h - h.min()) / max(float(h.max() - h.min()), 1e-12)
    colors = np.zeros((h.shape[0], h.shape[1], 4), dtype=np.uint8)
    colors[..., 0] = 255
    colors[..., 1] = (80 * (1.0 - h)).astype(np.uint8)
    colors[..., 3] = (150 * h).astype(np.uint8)
    image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(colors, mode="RGBA"))
    draw = ImageDraw.Draw(image)
    ys, xs = np.nonzero(box_mask)
    if len(xs):
        draw.rectangle((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), outline="lime", width=3)
    for x, y in gt_parts:
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="cyan")
    for x, y in pred_parts:
        draw.line((x - 4, y, x + 4, y), fill="yellow", width=2)
        draw.line((x, y - 4, x, y + 4), fill="yellow", width=2)
    image.convert("RGB").save(path, quality=95)


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
    loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(getattr(cfg.DATA, "PIN_MEMORY", True)),
        drop_last=False,
    )
    annotations = CUBAnnotations(find_cub_root(dataset_val))
    image_size = int(cfg.DATA.IMG_SIZE)
    crop = bool(cfg.TEST.CROP)
    samples_meta = dataset_val.samples
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
        aux = find_aux(output)
        part_attention = aux.get(f"part_attn_{args.layer}")
        if part_attention is None:
            raise KeyError(f"missing part_attn_{args.layer}; update MetaFG_meta.py to expose pre-dropout part attention")

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
            row = {
                "image_id": image_id,
                "image": relative_path,
                "target": int(targets[local_index]),
                "layer": args.layer,
                "foreground_kind": foreground_kind,
                "visible_parts": int(len(gt_parts)),
            }

            source_maps = {}
            for source in args.map_sources:
                if source == "part_attention":
                    token_map = part_maps.mean(axis=0)
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
            for part_map in part_maps:
                dense_part = token_vector_to_dense(part_map.reshape(-1), image_size)
                per_part_energy.append(foreground_energy_fraction(dense_part, foreground))
            row["part_foreground_energy_mean"] = float(np.nanmean(per_part_energy))
            row["part_foreground_energy_min"] = float(np.nanmin(per_part_energy))
            rows.append(row)

            if global_index < args.save_overlays:
                overlay_source = "curvature" if "curvature" in source_maps else next(iter(source_maps))
                save_overlay(
                    output_dir / f"overlay_{global_index:05d}.jpg",
                    input_tensor_to_pil(images[local_index]),
                    source_maps[overlay_source],
                    box_mask,
                    gt_parts,
                    pred_parts,
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


if __name__ == "__main__":
    main()
