"""

用法：
  python visualize_token.py --cfg /raid/MetaFormer-V8-Lite/output/MetaFG_meta_2/inat2018_meta/config.json --ckpt /raid/MetaFormer-V8-Lite/output/MetaFG_meta_2/inat2018_meta/best.pth  --out ./figs/inat2018 --num-images 20 --layer 2

CUDA_VISIBLE_DEVICES=2 python visualize_token.py --cfg /raid/viz/MetaFormer_viz/output/MetaFG_2/stanfordcars-5-5e-3/config.json --ckpt /raid/viz/MetaFormer_viz/output/MetaFG_2/stanfordcars-5-5e-3/best.pth  --out ./figs/cars_low --num-images 20 --layer 2

CUDA_VISIBLE_DEVICES=4 python visualize_token.py --cfg /raid/viz/MetaFormer_viz/output/MetaFG_meta_2/cub-200-vis/config.json --ckpt /raid/viz/MetaFormer_viz/output/MetaFG_meta_2/cub-200-vis/best.pth  --out ./figs/cub_test --num-images 10 --layer 2

Curv-Part publication visualization (top-conference style)
==========================================================

Design goals
------------
1) Main qualitative figures use a clean 1x4 layout.
2) Full continuous heatmaps are shown; we do NOT hide 98% of the map.
3) Token selection uses an exact configurable integer --topk (e.g. --topk 8),
   never a fixed percentage.
4) No CUB bounding-box dependency. This script contains NO --cub-bbox option.
5) Distillation fidelity is evaluated in the EXACT training space used by
   SemanticPartTokenGeneratorV6:
       student = zscore(student_raw)
       teacher = zscore(log1p(hvp_raw))
   instead of incorrectly comparing final softmax curvature weights with HVP.
6) Part-token visualization uses attn_raw (the actual attention used to create
   part tokens), NOT part_assign (a token->semantic-part similarity diagnostic).
7) Every input image gets its own output folder.

Main figures per image
----------------------
01_curvature_vs_gradient
    Input | Distilled curvature | Classification gradient | Top-K comparison

02_distillation_fidelity
    Input | HVP teacher | Student raw | |Teacher-Student| error

03_semantic_grounding
    Input | Category semantic | Token-part semantic similarity | Curvature

04_part_attention_groupXX
    Input | Part attention 1 | Part attention 2 | Part attention 3
    (actual attn_raw used to form part tokens; groups are ranked by concentration)

05_part_generation
    Input | Curvature | Student saliency | Union part attention + part centers

06_cross_stage
    Input | Stage-1 curvature | Stage-2 curvature | Cross-stage difference

Supplementary diagnostics
-------------------------
07_part_diversity
08_attribute_assignment

The visualization conventions intentionally follow common FGVC/ViT paper
practice: original image + compact small-multiple heatmaps + a small number of
selected patches, while correlations/overlaps are exported to CSV rather than
placing a large scatter plot in every qualitative figure.
"""

import argparse
import csv
import math
import os
import re
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from visualize import (
    _pm,
    load_config,
    unnormalize,
    build_model,
    build_loader,
)

import apex.amp


# -----------------------------------------------------------------------------
# Plot defaults
# -----------------------------------------------------------------------------

PAPER_DPI = 600
HEAT_CMAP = "turbo"
ERROR_CMAP = "magma"
MATRIX_CMAP = "RdBu_r"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def make_safe_filename(name):
    name = os.path.basename(str(name))
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name if name else "unknown_image"


def get_dataset_image_names(dataset):
    for attr in ("samples", "imgs"):
        if hasattr(dataset, attr):
            vals = getattr(dataset, attr)
            names = []
            for item in vals:
                path = item[0] if isinstance(item, (tuple, list)) else item
                names.append(os.path.basename(str(path)))
            if names:
                return names
    for attr in ("image_paths", "paths", "files"):
        if hasattr(dataset, attr):
            vals = getattr(dataset, attr)
            names = [os.path.basename(str(p)) for p in vals]
            if names:
                return names
    return None


def get_image_names_from_loader(loader):
    if not hasattr(loader, "dataset"):
        return None
    ds = loader.dataset
    names = get_dataset_image_names(ds)
    if names is not None:
        return names
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        base = get_dataset_image_names(ds.dataset)
        if base is not None:
            return [base[int(i)] for i in ds.indices]
    if hasattr(ds, "dataset"):
        return get_dataset_image_names(ds.dataset)
    return None


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flatten_score(x):
    if x is None:
        return None
    a = _to_numpy(x).astype(np.float32)
    a = np.squeeze(a)
    if a.ndim != 1:
        a = a.reshape(-1)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def token_grid(x):
    x = _flatten_score(x)
    n = x.size
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError("Token count %d is not a square grid." % n)
    return x.reshape(side, side)


def resize_grid(grid, h, w):
    t = torch.as_tensor(grid, dtype=torch.float32)[None, None]
    out = F.interpolate(t, size=(int(h), int(w)), mode="bilinear", align_corners=False)
    return out[0, 0].cpu().numpy()


def robust_norm(x, low=1.0, high=99.0):
    """Robust [0,1] normalization ONLY for visualization."""
    a = np.asarray(x, dtype=np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    lo = float(np.percentile(finite, low))
    hi = float(np.percentile(finite, high))
    if hi <= lo + 1e-12:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo + 1e-12:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def zscore_np(x, eps=1e-6):
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    mu = float(np.mean(a))
    sd = float(np.std(a))
    if sd < eps:
        return np.zeros_like(a, dtype=np.float64)
    return (a - mu) / sd


def _rankdata(a):
    """Average ranks for ties, scipy-free."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and a[order[j]] == a[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


def pearson_np(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / den)


def spearman_np(a, b):
    return pearson_np(_rankdata(a), _rankdata(b))


def topk_indices(score, k):
    s = np.asarray(score).reshape(-1)
    k = max(1, min(int(k), s.size))
    idx = np.argpartition(s, -k)[-k:]
    idx = idx[np.argsort(s[idx])[::-1]]
    return idx.astype(np.int64)


def topk_overlap(a, b, k):
    A = set(topk_indices(a, k).tolist())
    B = set(topk_indices(b, k).tolist())
    k_eff = max(1, min(int(k), len(A), len(B)))
    return len(A & B) / float(k_eff)


def save_csv(path, rows, fields):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def save_fig(fig, stem):
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    fig.savefig(stem + ".png", dpi=PAPER_DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem + ".pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def clean_axis(ax, title=None):
    if title is not None:
        ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def show_heat(ax, img, score, title, alpha=0.55, cmap=HEAT_CMAP):
    """Continuous full heatmap overlay (not sparse percentile masking)."""
    g = token_grid(score)
    g = robust_norm(g)
    up = resize_grid(g, img.shape[0], img.shape[1])
    ax.imshow(img)
    ax.imshow(up, cmap=cmap, vmin=0.0, vmax=1.0, alpha=float(alpha), interpolation="bilinear")
    clean_axis(ax, title)
    return g


def draw_topk_boxes(ax, img, score, k, edgecolor="red", linewidth=1.8, label=None):
    g = token_grid(score)
    h_grid, w_grid = g.shape
    h_img, w_img = img.shape[:2]
    ph = h_img / float(h_grid)
    pw = w_img / float(w_grid)
    idx = topk_indices(score, k)
    for token_id in idx:
        r = int(token_id // w_grid)
        c = int(token_id % w_grid)
        ax.add_patch(Rectangle(
            (c * pw, r * ph), pw, ph,
            fill=False, edgecolor=edgecolor, linewidth=linewidth
        ))
    if label:
        ax.plot([], [], color=edgecolor, linewidth=linewidth, label=label)
    return idx


def draw_topk_comparison(ax, img, a, b, k, a_name="Curvature", b_name="Gradient"):
    ax.imshow(img)
    ga = token_grid(a)
    h_grid, w_grid = ga.shape
    h_img, w_img = img.shape[:2]
    ph, pw = h_img / h_grid, w_img / w_grid
    A = set(topk_indices(a, k).tolist())
    B = set(topk_indices(b, k).tolist())
    for token_id in sorted(A | B):
        r, c = divmod(int(token_id), w_grid)
        if token_id in A and token_id in B:
            color, lw = "gold", 2.4
        elif token_id in A:
            color, lw = "red", 1.8
        else:
            color, lw = "cyan", 1.8
        ax.add_patch(Rectangle((c * pw, r * ph), pw, ph, fill=False,
                               edgecolor=color, linewidth=lw))
    ax.plot([], [], color="red", lw=2, label=a_name)
    ax.plot([], [], color="cyan", lw=2, label=b_name)
    ax.plot([], [], color="gold", lw=2.5, label="Overlap")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    clean_axis(ax)


# -----------------------------------------------------------------------------
# Model/output helpers
# -----------------------------------------------------------------------------

def unpack_model_output(output):
    logits = None
    aux = None
    attn1 = None
    attn2 = None
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, (tuple, list)):
        if len(output) == 4:
            logits, aux, attn1, attn2 = output
        else:
            for item in output:
                if logits is None and isinstance(item, torch.Tensor) and item.ndim == 2:
                    logits = item
                if aux is None and isinstance(item, dict):
                    aux = item
    if logits is None:
        raise RuntimeError("Cannot find [B,C] logits in model output.")
    return logits, aux, attn1, attn2


def load_checkpoint(model, state_dict, allow_partial=False):
    msg = model.load_state_dict(state_dict, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    if missing or unexpected:
        print("[checkpoint] missing=%d unexpected=%d" % (len(missing), len(unexpected)))
        if missing:
            print("  missing:", missing[:30])
        if unexpected:
            print("  unexpected:", unexpected[:30])
        if not allow_partial:
            raise RuntimeError(
                "Checkpoint mismatch. Use --allow-partial-ckpt only for debugging, not paper figures."
            )


def _dict_from_module_output(output):
    if isinstance(output, dict):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, dict):
                return item
    return None


def _register_part_aux_hooks(model, layers=(1, 2)):
    captured = {}
    handles = []
    for layer in layers:
        name = "part_gen_%d" % layer
        if not hasattr(model, name):
            continue
        module = getattr(model, name)
        def make_hook(lid):
            def hook(mod, inp, out):
                d = _dict_from_module_output(out)
                if d is not None:
                    captured[lid] = d
            return hook
        handles.append(module.register_forward_hook(make_hook(layer)))
    return captured, handles


def _restore_training_states(states):
    for module, training in states:
        module.train(training)


def _freeze_stochastic_modules(model):
    """Keep parent in train mode for HVP, but freeze unrelated stochastic layers."""
    for m in model.modules():
        cname = m.__class__.__name__.lower()
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
        elif isinstance(m, nn.modules.dropout._DropoutNd):
            m.eval()
        elif "droppath" in cname or "stochasticdepth" in cname:
            m.eval()


def eval_forward_with_internal_aux(model, samples, meta):
    captured, handles = _register_part_aux_hooks(model, layers=(1, 2))
    try:
        model.eval()
        with torch.no_grad():
            output = model(samples, meta, return_aux=True)
        logits, outer_aux, attn1, attn2 = unpack_model_output(output)
        return logits, outer_aux, attn1, attn2, captured
    finally:
        for h in handles:
            h.remove()


def hvp_forward_with_internal_aux(model, samples, meta, layer, seed=None):
    """
    One HVP assessment forward. Captures the exact internal student_raw/hvp_raw
    returned by SemanticPartTokenGeneratorV6.
    """
    states = [(m, bool(m.training)) for m in model.modules()]
    captured, handles = _register_part_aux_hooks(model, layers=(layer,))
    part_module = getattr(model, "part_gen_%d" % layer)
    try:
        model.train(True)
        _freeze_stochastic_modules(model)
        if hasattr(part_module, "set_hvp_seed"):
            part_module.set_hvp_seed(seed)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(samples, meta, return_aux=True)
        logits, outer_aux, attn1, attn2 = unpack_model_output(output)
        inner = captured.get(layer)
        if inner is None:
            raise RuntimeError("Could not capture internal aux from part_gen_%d." % layer)
        return logits, outer_aux, attn1, attn2, inner
    finally:
        if hasattr(part_module, "set_hvp_seed"):
            part_module.set_hvp_seed(None)
        for h in handles:
            h.remove()
        _restore_training_states(states)
        model.zero_grad(set_to_none=True)


def compute_ce_gradient(model, samples, meta, targets, layer):
    """Classification first-order gradient at part_gen_{layer}.input_proj output."""
    captured = {}
    module = getattr(model, "part_gen_%d" % layer)

    def hook(mod, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not isinstance(out, torch.Tensor):
            raise TypeError("input_proj output is not a tensor")
        captured["x"] = out

    h = module.input_proj.register_forward_hook(hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            output = model(samples, meta, return_aux=True)
            logits, _, _, _ = unpack_model_output(output)
            loss = F.cross_entropy(logits, targets)
            if "x" not in captured:
                raise RuntimeError("Gradient hook did not capture token features.")
            gx = torch.autograd.grad(loss, captured["x"], retain_graph=False, allow_unused=False)[0]
        return gx.norm(p=2, dim=-1).detach().cpu()
    finally:
        h.remove()
        model.zero_grad(set_to_none=True)


def batch_item(aux, key, i, squeeze=True):
    if aux is None or key not in aux or aux[key] is None:
        return None
    x = aux[key]
    if isinstance(x, torch.Tensor):
        x = x[i].detach().cpu().numpy()
    else:
        x = np.asarray(x)[i]
    if squeeze:
        x = np.squeeze(x)
    return x


# -----------------------------------------------------------------------------
# Exact distillation-space extraction
# -----------------------------------------------------------------------------

def collect_teacher_student_fidelity(model, samples, meta, layer, repeats=1, seed=0):
    """
    Training loss in the provided V6 code uses:
        s = zscore(student_raw)
        t = zscore(log1p(hvp_raw))
    We reproduce exactly that space here.

    Returns arrays shaped [B,N]:
        teacher_z_mean, student_train_z_mean, teacher_self_rho
    """
    repeats = max(1, int(repeats))
    teacher_z_runs = []
    student_z_runs = []
    for r in range(repeats):
        _, _, _, _, inner = hvp_forward_with_internal_aux(
            model, samples, meta, layer, seed=(None if seed is None else int(seed) + r)
        )
        sraw = inner.get("student_raw")
        hraw = inner.get("hvp_raw")
        if sraw is None or hraw is None:
            raise RuntimeError(
                "Internal aux must expose student_raw and hvp_raw for correct distillation visualization."
            )
        sraw = _to_numpy(sraw).astype(np.float64)
        hraw = _to_numpy(hraw).astype(np.float64)
        if hraw.ndim == 3 and hraw.shape[-1] == 1:
            hraw = hraw[..., 0]
        if sraw.ndim != 2 or hraw.ndim != 2 or sraw.shape != hraw.shape:
            raise RuntimeError("student_raw/hvp_raw shape mismatch: %s vs %s" % (sraw.shape, hraw.shape))
        s_z = np.stack([zscore_np(row) for row in sraw], axis=0)
        t_log = np.log1p(np.maximum(hraw, 0.0))
        t_z = np.stack([zscore_np(row) for row in t_log], axis=0)
        student_z_runs.append(s_z)
        teacher_z_runs.append(t_z)

    S = np.stack(student_z_runs, axis=0)  # R,B,N
    T = np.stack(teacher_z_runs, axis=0)
    s_mean = S.mean(axis=0)
    t_mean = T.mean(axis=0)

    reliability = np.full((T.shape[1],), np.nan, dtype=np.float64)
    if repeats > 1:
        for b in range(T.shape[1]):
            vals = []
            for r1 in range(repeats):
                for r2 in range(r1 + 1, repeats):
                    vals.append(spearman_np(T[r1, b], T[r2, b]))
            reliability[b] = float(np.nanmean(vals)) if vals else np.nan
    return t_mean, s_mean, reliability


# -----------------------------------------------------------------------------
# Publication figures: all main qualitative plots are 1x4
# -----------------------------------------------------------------------------
def plot_curvature_vs_gradient_split(img, curvature, gradient, stem, topk=8, alpha=0.55):
    """01：每个子面板单独存一张（Input / curvature / gradient / Top-K）。"""
    c = _flatten_score(curvature)
    g = _flatten_score(gradient)
    if c.size != g.size:
        raise ValueError("Curvature/gradient token mismatch: %d vs %d" % (c.size, g.size))

    p = pearson_np(c, g)
    r = spearman_np(c, g)
    ov = topk_overlap(c, g, topk)

    # 01a: 输入原图
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.imshow(img)
    clean_axis(ax, "Input")
    fig.tight_layout()
    save_fig(fig, stem + "_01a_input")

    # 01b: 曲率热力图
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    show_heat(ax, img, c, "Distilled curvature", alpha=alpha)
    fig.tight_layout()
    save_fig(fig, stem + "_01b_curvature")

    # 01c: 一阶梯度热力图
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    show_heat(ax, img, g, "Classification gradient", alpha=alpha)
    fig.tight_layout()
    save_fig(fig, stem + "_01c_gradient")

    # 01d: Top-K 对比框
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    draw_topk_comparison(ax, img, c, g, topk,
                         a_name="Curvature", b_name="Gradient")
    ax.set_title("Top-%d patches\nSpearman=%.3f | O@%d=%.3f" % (topk, r, topk, ov), fontsize=10)
    fig.tight_layout()
    save_fig(fig, stem + "_01d_topk")

    return {"pearson": p, "spearman": r, "topk_overlap": ov, "topk": int(topk)}


def _z_to_display(z):
    """Shared display mapping for teacher/student z-scores."""
    z = np.asarray(z, dtype=np.float32)
    z = np.clip(z, -2.5, 2.5)
    return (z + 2.5) / 5.0


def plot_distillation_1x4(img, teacher_z, student_z, stem, topk=8, alpha=0.55,
                          teacher_self_rho=float("nan")):
    t = _flatten_score(teacher_z)
    s = _flatten_score(student_z)
    if t.size != s.size:
        raise ValueError("Teacher/student token mismatch")

    p = pearson_np(t, s)
    r = spearman_np(t, s)
    ov = topk_overlap(t, s, topk)
    mae = float(np.mean(np.abs(t - s)))

    tg = token_grid(_z_to_display(t))
    sg = token_grid(_z_to_display(s))
    diff = token_grid(np.abs(t - s))
    diff = robust_norm(diff)

    h, w = img.shape[:2]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    axes[0].imshow(img)
    clean_axis(axes[0], "Input")

    axes[1].imshow(img)
    axes[1].imshow(resize_grid(tg, h, w), cmap=HEAT_CMAP, vmin=0, vmax=1,
                   alpha=alpha, interpolation="bilinear")
    clean_axis(axes[1], "HVP Teacher\nlog1p + z-score")

    axes[2].imshow(img)
    axes[2].imshow(resize_grid(sg, h, w), cmap=HEAT_CMAP, vmin=0, vmax=1,
                   alpha=alpha, interpolation="bilinear")
    clean_axis(axes[2], "Student\nraw head + z-score")

    axes[3].imshow(resize_grid(diff, h, w), cmap=ERROR_CMAP, vmin=0, vmax=1,
                   interpolation="bilinear")
    title = "Absolute z-error\nSpearman=%.3f | O@%d=%.3f" % (r, topk, ov)
    if np.isfinite(teacher_self_rho):
        title += "\nTeacher self-rho=%.3f" % teacher_self_rho
    clean_axis(axes[3], title)

    fig.suptitle("Online Curvature Distillation: Spatial Fidelity", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)
    return {
        "pearson": p,
        "spearman": r,
        "topk_overlap": ov,
        "z_mae": mae,
        "teacher_self_rho": float(teacher_self_rho),
        "topk": int(topk),
    }


def plot_semantic_grounding_1x4(img, category_score, token_semantic, curvature,
                                stem, alpha=0.55):
    cat = _flatten_score(category_score)
    sem = _flatten_score(token_semantic)
    curv = _flatten_score(curvature)
    if not (cat.size == sem.size == curv.size):
        raise ValueError("Semantic/curvature token count mismatch")

    r_cat = spearman_np(curv, cat)
    r_sem = spearman_np(curv, sem)

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    axes[0].imshow(img)
    clean_axis(axes[0], "Input")
    show_heat(axes[1], img, cat, "Category semantic score", alpha=alpha, cmap="Blues")
    show_heat(axes[2], img, sem, "Token-part semantic similarity", alpha=alpha, cmap="Oranges")
    show_heat(axes[3], img, curv,
              "Distilled curvature\nρ(cat)=%.3f | ρ(local)=%.3f" % (r_cat, r_sem),
              alpha=alpha, cmap=HEAT_CMAP)
    fig.suptitle("Hierarchical Semantic Grounding and Curvature", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)
    return {"curv_category_spearman": r_cat, "curv_local_semantic_spearman": r_sem}


def _part_concentration_scores(attn_raw):
    A = np.asarray(attn_raw, dtype=np.float64)
    A = np.maximum(A, 0.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-12)
    n = A.shape[1]
    entropy = -(A * np.log(A + 1e-12)).sum(axis=1) / math.log(max(n, 2))
    peak_ratio = n * A.max(axis=1)
    # high peak + low entropy = concentrated/localized
    score = peak_ratio + (1.0 - entropy)
    return score, entropy, peak_ratio

def plot_part_attention_separate(img, attn_raw, out_dir, topk=8, alpha=0.60):
    """04：每个 part query 单独存一张（不再每 3 个合并成 1x4）。"""
    A = np.asarray(attn_raw, dtype=np.float32)
    if A.ndim != 2:
        raise ValueError("attn_raw must be [P,N], got %s" % (A.shape,))
    p, n = A.shape
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError("attn_raw token count is not square")

    score, entropy, peak_ratio = _part_concentration_scores(A)
    order = np.argsort(score)[::-1]   # 按集中度降序

    for rank, q in enumerate(order):
        q = int(q)
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        show_heat(
            ax, img, A[q],
            "Part %d attention\npeak×N=%.2f | H=%.2f" % (q + 1, peak_ratio[q], entropy[q]),
            alpha=alpha, cmap=HEAT_CMAP
        )
        draw_topk_boxes(ax, img, A[q], topk, edgecolor="red", linewidth=1.0)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, "04_part_attention_p%02d_rank%02d" % (q + 1, rank + 1)))
    return order, entropy, peak_ratio


def part_centers(attn_raw):
    A = np.asarray(attn_raw, dtype=np.float64)
    p, n = A.shape
    side = int(round(math.sqrt(n)))
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    coords = np.stack([yy.reshape(-1), xx.reshape(-1)], axis=1)
    centers = []
    for q in range(p):
        w = np.maximum(A[q], 0.0)
        w = w / (w.sum() + 1e-12)
        c = (w[:, None] * coords).sum(axis=0)
        centers.append(c)
    return np.asarray(centers), side


def plot_part_generation_1x4(img, curvature, student_saliency, attn_raw, stem, alpha=0.55):
    curv = _flatten_score(curvature)
    sal = _flatten_score(student_saliency)
    A = np.asarray(attn_raw, dtype=np.float32)
    if A.ndim != 2 or A.shape[1] != curv.size or sal.size != curv.size:
        raise ValueError("part generation shape mismatch")

    Ac = np.clip(A, 0.0, 1.0)
    union = 1.0 - np.prod(1.0 - Ac, axis=0)
    centers, side = part_centers(A)

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    axes[0].imshow(img)
    clean_axis(axes[0], "Input")
    show_heat(axes[1], img, curv, "Distilled curvature\nwhere-to-look", alpha=alpha)
    show_heat(axes[2], img, sal, "Student saliency\nshared foreground gate", alpha=alpha, cmap="viridis")
    show_heat(axes[3], img, union, "Union part attention\n+ weighted part centers", alpha=alpha)

    h, w = img.shape[:2]
    ph, pw = h / float(side), w / float(side)
    for q, (rr, cc) in enumerate(centers):
        x = (cc + 0.5) * pw
        y = (rr + 0.5) * ph
        axes[3].scatter([x], [y], s=28, facecolors="white", edgecolors="red", linewidths=1.2)
        axes[3].text(x + 2, y - 2, "P%d" % (q + 1), color="red", fontsize=7,
                     fontweight="bold")

    fig.suptitle("From Curvature to Part-Token Generation", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)
    return {"num_parts": int(A.shape[0])}


def plot_cross_stage_1x4(img, stage1_curv, stage2_curv, stem, alpha=0.55):
    c1 = _flatten_score(stage1_curv)
    c2 = _flatten_score(stage2_curv)
    g1 = token_grid(c1)
    g2 = token_grid(c2)
    g2_up = resize_grid(g2, g1.shape[0], g1.shape[1])

    n1 = robust_norm(g1)
    n2 = robust_norm(g2)
    n2_up = robust_norm(g2_up)
    diff = np.abs(n1 - n2_up)
    rho = spearman_np(n1.reshape(-1), n2_up.reshape(-1))

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    axes[0].imshow(img)
    clean_axis(axes[0], "Input")
    show_heat(axes[1], img, c1, "Stage 1 curvature", alpha=alpha)
    show_heat(axes[2], img, c2, "Stage 2 curvature", alpha=alpha)
    axes[3].imshow(resize_grid(diff, img.shape[0], img.shape[1]), cmap=ERROR_CMAP,
                   vmin=0, vmax=1, interpolation="bilinear")
    clean_axis(axes[3], "Cross-stage difference\nSpearman=%.3f" % rho)
    fig.suptitle("Cross-Stage Curvature Localization", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)
    return {"cross_stage_spearman": rho}


# -----------------------------------------------------------------------------
# Supplementary diagnostics
# -----------------------------------------------------------------------------

def compute_part_diversity_attn(attn_raw, topk=8):
    A = np.asarray(attn_raw, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("attn_raw must be [P,N]")
    P, N = A.shape
    A = np.maximum(A, 0.0)
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    cosine = An @ An.T
    sets = [set(topk_indices(A[p], topk).tolist()) for p in range(P)]
    jaccard = np.eye(P, dtype=np.float64)
    for i in range(P):
        for j in range(i + 1, P):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            v = inter / float(max(union, 1))
            jaccard[i, j] = jaccard[j, i] = v
    mask = ~np.eye(P, dtype=bool)
    coverage = len(set().union(*sets)) / float(N)
    return {
        "cosine": cosine,
        "jaccard": jaccard,
        "mean_cosine": float(cosine[mask].mean()) if P > 1 else np.nan,
        "mean_jaccard": float(jaccard[mask].mean()) if P > 1 else np.nan,
        "coverage": float(coverage),
    }


def plot_part_diversity(attn_raw, stem, topk=8):
    d = compute_part_diversity_attn(attn_raw, topk=topk)
    jac, cos = d["jaccard"], d["cosine"]
    P = jac.shape[0]
    labels = ["P%d" % (i + 1) for i in range(P)]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    im0 = axes[0].imshow(jac, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Top-%d spatial Jaccard" % topk)
    axes[0].set_xticks(range(P)); axes[0].set_yticks(range(P))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(labels, fontsize=8)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(cos, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Full-map cosine similarity")
    axes[1].set_xticks(range(P)); axes[1].set_yticks(range(P))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(labels, fontsize=8)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    vals = [d["mean_jaccard"], d["mean_cosine"], d["coverage"]]
    axes[2].bar(["Mean\nJaccard", "Mean\nCosine", "Union\nCoverage"], vals)
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Part diversity diagnostics")
    for i, v in enumerate(vals):
        axes[2].text(i, min(v + 0.025, 0.98), "%.3f" % v, ha="center", fontsize=9)

    fig.suptitle("Part-Token Diversity (computed on actual attn_raw)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)
    return d


def plot_attribute_assignment(attr_attn, stem):
    A = np.asarray(attr_attn, dtype=np.float64)
    A = np.squeeze(A)
    if A.ndim != 2:
        raise ValueError("attr_attn must be [P,A]")
    P, M = A.shape
    uniform = 1.0 / float(M)
    centered = A - uniform
    max_abs = max(float(np.max(np.abs(centered))), 1e-6)

    probs = np.clip(A, 1e-12, None)
    probs = probs / probs.sum(axis=1, keepdims=True)
    entropy = -(probs * np.log(probs)).sum(axis=1) / math.log(max(M, 2))
    peak_ratio = probs.max(axis=1) / uniform

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    im = axes[0].imshow(centered, cmap=MATRIX_CMAP, vmin=-max_abs, vmax=max_abs, aspect="auto")
    axes[0].set_title("Assignment deviation from uniform")
    axes[0].set_xlabel("Attribute index")
    axes[0].set_ylabel("Part query")
    axes[0].set_yticks(range(P)); axes[0].set_yticklabels(["P%d" % (i+1) for i in range(P)])
    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)

    axes[1].bar(np.arange(P), entropy)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Normalized attribute entropy")
    axes[1].set_xlabel("Part query")
    axes[1].set_xticks(range(P)); axes[1].set_xticklabels(["P%d" % (i+1) for i in range(P)], rotation=45)

    axes[2].bar(np.arange(P), peak_ratio)
    axes[2].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[2].set_title("Peak / uniform probability")
    axes[2].set_xlabel("Part query")
    axes[2].set_xticks(range(P)); axes[2].set_xticklabels(["P%d" % (i+1) for i in range(P)], rotation=45)

    fig.suptitle("Part-Attribute Assignment Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)
    return {
        "attr_entropy_mean": float(np.mean(entropy)),
        "attr_peak_ratio_mean": float(np.mean(peak_ratio)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="./figs_topconf")
    ap.add_argument("--num-images", type=int, default=16)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--layer", type=int, default=2, choices=[1, 2])
    ap.add_argument("--topk", type=int, default=8,
                    help="Exact number of selected image tokens. Example: --topk 8")
    ap.add_argument("--teacher-repeats", type=int, default=1,
                    help="Number of HVP assessment forwards to average. Model already averages hvp_samples internally.")
    ap.add_argument("--heat-alpha", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-teacher", action="store_true")
    ap.add_argument("--allow-partial-ckpt", action="store_true")
    args = ap.parse_args()

    if args.topk < 1:
        raise ValueError("--topk must be >= 1")
    if not (0.0 < args.heat_alpha <= 1.0):
        raise ValueError("--heat-alpha must be in (0,1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out, exist_ok=True)
    aggregate_dir = os.path.join(args.out, "_aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    cfg = load_config(args.cfg)
    cfg.defrost()
    cfg.EVAL_MODE = True
    cfg.MODEL.assess = True
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[info] device =", device)

    if not hasattr(apex.amp, "_amp_state"):
        apex.amp._amp_state = type("DummyState", (), {})()
    if not hasattr(apex.amp._amp_state, "handle") or apex.amp._amp_state.handle is None:
        apex.amp._amp_state.handle = type("DummyHandle", (), {"_is_active": False})()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, pickle_module=_pm)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in sd.items()}

    hw = sd.get("head.weight")
    if hw is not None:
        cfg.defrost()
        cfg.MODEL.NUM_CLASSES = int(hw.shape[0])
        cfg.freeze()

    model = build_model(cfg)
    load_checkpoint(model, sd, allow_partial=args.allow_partial_ckpt)
    model = model.to(device).eval()
    model.assess = True

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29503")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo")

    _, _, _, loader, _ = build_loader(cfg)
    image_names = get_image_names_from_loader(loader)
    print("[info] image names:", "found %d" % len(image_names) if image_names is not None else "not found")

    manifest = []
    curv_grad_rows = []
    distill_rows = []
    semantic_rows = []
    diversity_rows = []
    attr_rows = []
    cross_rows = []

    n_done = 0
    dataset_index = 0
    shapes_printed = False

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches or n_done >= args.num_images:
            break

        if cfg.DATA.ADD_META:
            samples, targets, meta = batch
            meta = torch.stack([m.float() for m in meta], dim=0).to(device)
        else:
            samples, targets = batch
            meta = None

        samples = samples.to(device)
        targets = targets.to(device)
        B = samples.shape[0]

        # ---------------------------------------------------------
        # Eval path: capture internal aux from BOTH part generators.
        # ---------------------------------------------------------
        logits, outer_aux, attn1, attn2, inner_eval = eval_forward_with_internal_aux(
            model, samples, meta
        )

        # Classification gradient is intentionally a separate first-order
        # diagnostic. Its heatmap is robust-normalized for visibility.
        grad_batch = compute_ce_gradient(model, samples, meta, targets, args.layer)

        # ---------------------------------------------------------
        # HVP teacher fidelity in the EXACT training distillation space.
        # ---------------------------------------------------------
        teacher_z_batch = None
        student_train_z_batch = None
        teacher_rel_batch = None
        if not args.no_teacher:
            try:
                teacher_z_batch, student_train_z_batch, teacher_rel_batch = collect_teacher_student_fidelity(
                    model, samples, meta, args.layer,
                    repeats=args.teacher_repeats, seed=args.seed + batch_idx * 100
                )
            except Exception as e:
                print("[warning] teacher/student fidelity skipped for batch %d: %r" % (batch_idx, e))

        # Eval student raw is captured directly from the module.
        selected_inner = inner_eval.get(args.layer)
        if selected_inner is None:
            raise RuntimeError("No internal aux captured for part_gen_%d." % args.layer)

        if not shapes_printed:
            shapes_printed = True
            print("\n[shape validation: selected part_gen_%d]" % args.layer)
            for key in [
                "student_raw", "curvature", "curv_weight", "hvp_raw",
                "part_assign", "attn_raw", "query_assignment", "student_saliency",
                "per_token_sim", "attr_attn", "s_cls"
            ]:
                v = selected_inner.get(key)
                if v is None:
                    print("  %-20s : None" % key)
                elif isinstance(v, torch.Tensor):
                    print("  %-20s : %s" % (key, tuple(v.shape)))
                else:
                    print("  %-20s : %s" % (key, type(v).__name__))
            print("  IMPORTANT: distillation figure uses student_raw vs log1p(hvp_raw), matching training.\n")

        for i in range(B):
            if n_done >= args.num_images:
                break

            img = unnormalize(samples[i].detach().cpu())
            img = np.asarray(img)
            original_name = (
                image_names[dataset_index]
                if image_names is not None and dataset_index < len(image_names)
                else "img_%05d.jpg" % dataset_index
            )
            safe = make_safe_filename(original_name)
            image_dir = os.path.join(args.out, "%04d_%s" % (dataset_index + 1, safe))
            os.makedirs(image_dir, exist_ok=True)
            plt.imsave(os.path.join(image_dir, "00_input.png"), img)

            inner = inner_eval.get(args.layer)
            if inner is None:
                raise RuntimeError("Missing internal aux for selected layer")

            # Correct meanings from V6:
            # student_raw: raw prediction head score
            # curv_weight: final positive bounded curvature weight used by classifier path
            # curvature: curv_weight[...,None]
            student_raw_eval = batch_item(inner, "student_raw", i)
            curvature = batch_item(inner, "curv_weight", i)
            if curvature is None:
                curvature = batch_item(inner, "curvature", i)
            gradient = _flatten_score(grad_batch[i])
            s_cls = batch_item(inner, "s_cls", i)
            per_token_sim = batch_item(inner, "per_token_sim", i)
            student_saliency = batch_item(inner, "student_saliency", i)
            attn_raw = batch_item(inner, "attn_raw", i, squeeze=False)
            attr_attn = batch_item(inner, "attr_attn", i, squeeze=False)

            curvature = _flatten_score(curvature)
            student_raw_eval = _flatten_score(student_raw_eval)
            s_cls = _flatten_score(s_cls)
            per_token_sim = _flatten_score(per_token_sim)
            student_saliency = _flatten_score(student_saliency)
            attn_raw = np.squeeze(np.asarray(attn_raw, dtype=np.float32)) if attn_raw is not None else None
            attr_attn = np.squeeze(np.asarray(attr_attn, dtype=np.float32)) if attr_attn is not None else None

            n_tokens = curvature.size
            k = min(args.topk, n_tokens)

            # 01: desired full-heatmap style, clean 1x4.
            m = plot_curvature_vs_gradient_split(
                img, curvature, gradient,
                os.path.join(image_dir, "01_curvature_vs_gradient"),
                topk=k, alpha=args.heat_alpha
            )
            curv_grad_rows.append({"image": original_name, "layer": args.layer, **m})

            # 02: EXACT training-space teacher/student fidelity.
            if teacher_z_batch is not None and student_raw_eval is not None:
                teacher_z = teacher_z_batch[i]
                student_eval_z = zscore_np(student_raw_eval)
                teacher_self = (
                    float(teacher_rel_batch[i])
                    if teacher_rel_batch is not None and i < len(teacher_rel_batch)
                    else float("nan")
                )
                dm = plot_distillation_1x4(
                    img, teacher_z, student_eval_z,
                    os.path.join(image_dir, "02_distillation_fidelity"),
                    topk=k, alpha=args.heat_alpha,
                    teacher_self_rho=teacher_self
                )
                dm.update({"image": original_name, "layer": args.layer})
                if student_train_z_batch is not None:
                    dm["train_spearman"] = spearman_np(teacher_z, student_train_z_batch[i])
                    dm["train_topk_overlap"] = topk_overlap(teacher_z, student_train_z_batch[i], k)
                else:
                    dm["train_spearman"] = float("nan")
                    dm["train_topk_overlap"] = float("nan")
                distill_rows.append(dm)

            # 03: semantic grounding 1x4.
            if s_cls is not None and per_token_sim is not None:
                sm = plot_semantic_grounding_1x4(
                    img, s_cls, per_token_sim, curvature,
                    os.path.join(image_dir, "03_semantic_grounding"),
                    alpha=args.heat_alpha
                )
                sm.update({"image": original_name, "layer": args.layer})
                semantic_rows.append(sm)

            # 04: actual part-token attentions, grouped as 1x4 rows.
            if attn_raw is not None and attn_raw.ndim == 2:
                plot_part_attention_separate(
                    img, attn_raw, image_dir, topk=k, alpha=args.heat_alpha
                )

                # 05: actual generation path.
                if student_saliency is not None:
                    plot_part_generation_1x4(
                        img, curvature, student_saliency, attn_raw,
                        os.path.join(image_dir, "05_part_generation"),
                        alpha=args.heat_alpha
                    )

                # 07: diversity computed on actual attn_raw, not part_assign.
                div = plot_part_diversity(
                    attn_raw, os.path.join(image_dir, "07_part_diversity"), topk=k
                )
                diversity_rows.append({
                    "image": original_name,
                    "layer": args.layer,
                    "topk": k,
                    "mean_jaccard": div["mean_jaccard"],
                    "mean_cosine": div["mean_cosine"],
                    "coverage": div["coverage"],
                })

            # 06: cross-stage full heatmaps, if both stage internals were captured.
            if 1 in inner_eval and 2 in inner_eval:
                c1 = batch_item(inner_eval[1], "curv_weight", i)
                c2 = batch_item(inner_eval[2], "curv_weight", i)
                if c1 is not None and c2 is not None:
                    cm = plot_cross_stage_1x4(
                        img, c1, c2,
                        os.path.join(image_dir, "06_cross_stage"),
                        alpha=args.heat_alpha
                    )
                    cross_rows.append({"image": original_name, **cm})

            # 08: attribute assignment diagnostic, no misleading 0.03 annotations.
            if attr_attn is not None and attr_attn.ndim == 2:
                am = plot_attribute_assignment(
                    attr_attn, os.path.join(image_dir, "08_attribute_assignment")
                )
                am.update({"image": original_name, "layer": args.layer})
                attr_rows.append(am)

            pred = int(torch.argmax(logits[i]).item())
            target = int(targets[i].item())
            manifest.append({
                "index": dataset_index + 1,
                "image": original_name,
                "folder": os.path.basename(image_dir),
                "prediction": pred,
                "target": target,
                "correct": int(pred == target),
            })

            print("[save] %s  topk=%d  pred=%d target=%d" % (image_dir, k, pred, target))
            dataset_index += 1
            n_done += 1

    # -------------------------------------------------------------------------
    # Aggregate CSVs
    # -------------------------------------------------------------------------
    save_csv(
        os.path.join(aggregate_dir, "manifest.csv"), manifest,
        ["index", "image", "folder", "prediction", "target", "correct"]
    )
    save_csv(
        os.path.join(aggregate_dir, "curvature_gradient_metrics.csv"), curv_grad_rows,
        ["image", "layer", "topk", "pearson", "spearman", "topk_overlap"]
    )
    save_csv(
        os.path.join(aggregate_dir, "distillation_metrics.csv"), distill_rows,
        ["image", "layer", "topk", "pearson", "spearman", "topk_overlap", "z_mae",
         "teacher_self_rho", "train_spearman", "train_topk_overlap"]
    )
    save_csv(
        os.path.join(aggregate_dir, "semantic_metrics.csv"), semantic_rows,
        ["image", "layer", "curv_category_spearman", "curv_local_semantic_spearman"]
    )
    save_csv(
        os.path.join(aggregate_dir, "part_diversity_metrics.csv"), diversity_rows,
        ["image", "layer", "topk", "mean_jaccard", "mean_cosine", "coverage"]
    )
    save_csv(
        os.path.join(aggregate_dir, "attribute_assignment_metrics.csv"), attr_rows,
        ["image", "layer", "attr_entropy_mean", "attr_peak_ratio_mean"]
    )
    save_csv(
        os.path.join(aggregate_dir, "cross_stage_metrics.csv"), cross_rows,
        ["image", "cross_stage_spearman"]
    )

    print("\n================ DONE ================")
    print("Output:", os.path.abspath(args.out))
    print("Main qualitative figures are 1x4.")
    print("Top-K is an exact token count: --topk %d" % args.topk)
    print("No CUB bounding-box code/options are used in this script.")
    print("Distillation fidelity uses EXACT training variables:")
    print("  student = zscore(student_raw)")
    print("  teacher = zscore(log1p(hvp_raw))")
    print("Part visualization/diversity uses actual attn_raw used to form part tokens.")
    print("======================================")


if __name__ == "__main__":
    main()
