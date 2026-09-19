"""
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
       student = mean-normalized predicted curvature
       teacher = mean-normalized HVP curvature
   instead of incorrectly comparing final softmax curvature weights with HVP.
6) Part-token visualization uses attn_raw (the actual attention used to create
   part tokens), NOT part_assign (a token->semantic-part similarity diagnostic).
7) Every input image gets its own output folder.

Main figures per image
----------------------
01_curvature_vs_gradient
    Input | Distilled curvature | Classification gradient | Top-K comparison

02_distillation_fidelity
    Input | HVP teacher | Student curvature | |Teacher-Student| error

03_semantic_grounding
    Input | Category semantic | Token-part semantic similarity | Curvature

04_part_attention_groupXX
    Input | Part attention 1 | Part attention 2 | Part attention 3
    (actual attn_raw used to form part tokens; groups are ranked by concentration)

05_part_generation
    Input | Curvature | Token-part semantic similarity | Mean part attention + part centers

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
import traceback

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
DISTILL_CMAP = "magma"
ERROR_CMAP = "inferno"
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
    """One selected-stage HVP assessment forward in eval mode.

    Key fixes over v4:
      1) request HVP only for the selected stage (avoids computing the very large
         stage-1 HVP when visualizing stage 2);
      2) use torch.random.fork_rng for deterministic probes instead of a custom
         CPU torch.Generator, which is more compatible with older PyTorch builds;
      3) keep BN/Dropout/DropPath in eval mode.
    """
    captured, handles = _register_part_aux_hooks(model, layers=(layer,))
    try:
        model.eval()
        model.zero_grad(set_to_none=True)

        def _run():
            # New MetaFG v4.1 supports force_hvp_layer.  Fall back to the older
            # modified MetaFG only if the keyword is genuinely unsupported.
            try:
                return model(
                    samples, meta, return_aux=True,
                    force_hvp=True, force_hvp_layer=int(layer)
                )
            except TypeError as e:
                msg = str(e)
                if 'force_hvp_layer' not in msg and 'unexpected keyword' not in msg:
                    raise
                return model(samples, meta, return_aux=True, force_hvp=True)

        with torch.enable_grad():
            if seed is None:
                output = _run()
            else:
                devices = []
                if samples.is_cuda:
                    devices = [samples.device.index]
                with torch.random.fork_rng(devices=devices, enabled=True):
                    torch.manual_seed(int(seed))
                    if samples.is_cuda:
                        torch.cuda.manual_seed_all(int(seed))
                    output = _run()

        logits, outer_aux, attn1, attn2 = unpack_model_output(output)
        inner = captured.get(layer)
        if inner is None:
            raise RuntimeError("Could not capture internal aux from part_gen_%d." % layer)
        if inner.get("hvp_curvature") is None:
            raise RuntimeError(
                "part_gen_%d ran with force_hvp=True but hvp_curvature is None. "
                "Check that SemanticPartTokenGeneratorV6.forward has force_hvp and "
                "that MetaFG forwards force_hvp to the selected part generator." % layer
            )
        return logits, outer_aux, attn1, attn2, inner
    finally:
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)

def compute_first_order_importance(
    model,
    samples,
    meta,
    targets,
    layer,
    objective="logit",
    class_source="pred",
    signal="grad",
):
    """
    First-order token importance at part_gen_{layer}.input_proj output.

    Why logit is the paper default:
      - Cross-entropy gradients can become extremely small/noisy on highly
        confident correct predictions.
      - Saliency visualizations usually differentiate a class score/logit,
        not the already-saturated CE loss.

    objective:
        "logit" -> d y_c / d x_i   (recommended qualitative baseline)
        "ce"    -> d CE / d x_i    (kept as an ablation/debug baseline)

    class_source (only for objective="logit"):
        "pred"   -> predicted class logit
        "target" -> ground-truth class logit

    signal:
        "grad"       -> ||grad_i||_2
        "gradxinput" -> ||grad_i * x_i||_2
    """
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
            if "x" not in captured:
                raise RuntimeError("Gradient hook did not capture token features.")

            if objective == "ce":
                scalar = F.cross_entropy(logits, targets)
            elif objective == "logit":
                if class_source == "target":
                    class_idx = targets
                elif class_source == "pred":
                    class_idx = logits.detach().argmax(dim=1)
                else:
                    raise ValueError("class_source must be 'pred' or 'target'")
                scalar = logits.gather(1, class_idx.view(-1, 1)).sum()
            else:
                raise ValueError("objective must be 'logit' or 'ce'")

            gx = torch.autograd.grad(
                scalar, captured["x"], retain_graph=False, allow_unused=False
            )[0]

            if signal == "grad":
                importance = gx.norm(p=2, dim=-1)
            elif signal == "gradxinput":
                importance = (gx * captured["x"]).norm(p=2, dim=-1)
            else:
                raise ValueError("signal must be 'grad' or 'gradxinput'")

        return importance.detach().cpu()
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


def recover_attn_raw(inner, i):
    """Return the actual [P,N] attention used to build part tokens.

    In the uploaded/current model this quantity is simply
        attn_raw = softmax(attn_logits, dim=-1)
    before attention dropout.  We intentionally do NOT reconstruct an r17/r18
    query_assignment/student_saliency path because that path does not exist in
    the checkpointed architecture uploaded by the user.
    """
    for key in ("attn_raw", "part_attn"):
        a = batch_item(inner, key, i, squeeze=False)
        if a is not None:
            a = np.squeeze(np.asarray(a, dtype=np.float32))
            if a.ndim == 2:
                return a, "direct %s" % key
    return None, "attn_raw/part_attn missing from internal aux"


# -----------------------------------------------------------------------------
# Exact distillation-space extraction
# -----------------------------------------------------------------------------


def _snapshot_bn_state(model):
    state = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            state.append((
                m,
                None if m.running_mean is None else m.running_mean.detach().clone(),
                None if m.running_var is None else m.running_var.detach().clone(),
                None if m.num_batches_tracked is None else m.num_batches_tracked.detach().clone(),
                m.momentum,
            ))
    return state


def _restore_bn_state(state):
    with torch.no_grad():
        for m, mean, var, nbt, momentum in state:
            if mean is not None and m.running_mean is not None:
                m.running_mean.copy_(mean)
            if var is not None and m.running_var is not None:
                m.running_var.copy_(var)
            if nbt is not None and m.num_batches_tracked is not None:
                m.num_batches_tracked.copy_(nbt)
            m.momentum = momentum


def trainmode_hvp_forward_with_internal_aux(model, samples, meta, layer, seed=None):
    """Reproduce the *training-time* teacher/student pair for distillation fidelity.

    This is intentionally different from ``eval()+force_hvp``.  The student was
    optimized against HVP targets produced while the network was in training mode,
    so a fidelity figure must evaluate the same stochastic feature distribution.

    To keep the visualization side-effect free we:
      - snapshot/restore BatchNorm running statistics;
      - temporarily set BN momentum to 0 (train-time batch statistics are still used);
      - disable HVP on the unselected part-generator only (HVP is detached and does
        not affect the forward features, so this only saves memory);
      - restore all module states after the forward.
    """
    captured, handles = _register_part_aux_hooks(model, layers=(layer,))
    was_training = model.training
    bn_state = _snapshot_bn_state(model)

    pg1 = getattr(model, 'part_gen_1', None)
    pg2 = getattr(model, 'part_gen_2', None)
    old_enable_1 = None if pg1 is None else getattr(pg1, 'enable_hvp', None)
    old_enable_2 = None if pg2 is None else getattr(pg2, 'enable_hvp', None)

    try:
        model.train(True)
        # Preserve BN running statistics while still using train-mode batch stats.
        for m, *_ in bn_state:
            m.momentum = 0.0

        if pg1 is not None and hasattr(pg1, 'enable_hvp'):
            pg1.enable_hvp = bool(int(layer) == 1)
        if pg2 is not None and hasattr(pg2, 'enable_hvp'):
            pg2.enable_hvp = bool(int(layer) == 2)

        model.zero_grad(set_to_none=True)

        def _run():
            # No force_hvp here: training=True is the actual condition used during
            # optimization.  This deliberately mirrors the checkpoint's training path.
            return model(samples, meta, return_aux=True)

        with torch.enable_grad():
            if seed is None:
                output = _run()
            else:
                devices = [samples.device.index] if samples.is_cuda else []
                with torch.random.fork_rng(devices=devices, enabled=True):
                    torch.manual_seed(int(seed))
                    if samples.is_cuda:
                        torch.cuda.manual_seed_all(int(seed))
                    output = _run()

        logits, outer_aux, attn1, attn2 = unpack_model_output(output)
        inner = captured.get(layer)
        if inner is None:
            raise RuntimeError('Could not capture internal aux from part_gen_%d.' % layer)
        if inner.get('hvp_curvature') is None:
            raise RuntimeError(
                'Training-mode part_gen_%d produced no HVP curvature. Check enable_hvp.' % layer
            )
        return logits, outer_aux, attn1, attn2, inner
    finally:
        if pg1 is not None and old_enable_1 is not None:
            pg1.enable_hvp = old_enable_1
        if pg2 is not None and old_enable_2 is not None:
            pg2.enable_hvp = old_enable_2
        _restore_bn_state(bn_state)
        model.train(was_training)
        for h in handles:
            h.remove()
        model.zero_grad(set_to_none=True)


def collect_teacher_student_fidelity(
    model, samples, meta, layer, repeats=4, seed=0, microbatch=2, topk=8
):
    """Training-time curvature distillation fidelity.

    Returns one deterministic reference Teacher/Student pair (first seed) for the
    1x4 qualitative figure plus per-image mean/std fidelity over repeated seeds.
    Metrics are computed on the actual mean-normalized training tensors.  The plot
    later converts them to percentile-rank maps only for spatial display.
    """
    repeats = max(1, int(repeats))
    microbatch = max(1, int(microbatch))
    B = int(samples.shape[0])

    teacher_ref_chunks = []
    student_ref_chunks = []
    rho_mean_chunks, rho_std_chunks = [], []
    ov_mean_chunks, ov_std_chunks = [], []
    mae_mean_chunks = []

    for b0 in range(0, B, microbatch):
        b1 = min(B, b0 + microbatch)
        sx = samples[b0:b1]
        mx = None if meta is None else meta[b0:b1]

        teacher_runs, student_runs = [], []
        for r in range(repeats):
            run_seed = None if seed is None else int(seed) + 1000 * b0 + r
            _, _, _, _, inner = trainmode_hvp_forward_with_internal_aux(
                model, sx, mx, layer, seed=run_seed
            )
            teacher = inner.get('hvp_curvature')
            student = inner.get('student_curvature_reg')
            if student is None:
                student = inner.get('student_curvature')
            if teacher is None or student is None:
                raise RuntimeError(
                    'Internal aux must expose hvp_curvature and student_curvature(_reg).'
                )

            teacher = _to_numpy(teacher).astype(np.float64)
            student = _to_numpy(student).astype(np.float64)
            if teacher.ndim == 3 and teacher.shape[-1] == 1:
                teacher = teacher[..., 0]
            if student.ndim == 3 and student.shape[-1] == 1:
                student = student[..., 0]
            if teacher.ndim != 2 or student.ndim != 2 or teacher.shape != student.shape:
                raise RuntimeError('student/teacher shape mismatch: %s vs %s' %
                                   (student.shape, teacher.shape))
            teacher_runs.append(teacher)
            student_runs.append(student)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        T = np.stack(teacher_runs, axis=0)  # R,b,N
        S = np.stack(student_runs, axis=0)  # R,b,N
        teacher_ref_chunks.append(T[0])
        student_ref_chunks.append(S[0])

        rho_m, rho_s, ov_m, ov_s, mae_m = [], [], [], [], []
        for bi in range(T.shape[1]):
            rhos = [spearman_np(T[r, bi], S[r, bi]) for r in range(repeats)]
            ovs = [topk_overlap(T[r, bi], S[r, bi], min(int(topk), T.shape[-1]))
                   for r in range(repeats)]
            maes = [float(np.mean(np.abs(T[r, bi] - S[r, bi]))) for r in range(repeats)]
            rho_m.append(float(np.nanmean(rhos)))
            rho_s.append(float(np.nanstd(rhos)))
            ov_m.append(float(np.nanmean(ovs)))
            ov_s.append(float(np.nanstd(ovs)))
            mae_m.append(float(np.nanmean(maes)))
        rho_mean_chunks.append(np.asarray(rho_m))
        rho_std_chunks.append(np.asarray(rho_s))
        ov_mean_chunks.append(np.asarray(ov_m))
        ov_std_chunks.append(np.asarray(ov_s))
        mae_mean_chunks.append(np.asarray(mae_m))

    return (
        np.concatenate(teacher_ref_chunks, axis=0),
        np.concatenate(student_ref_chunks, axis=0),
        np.concatenate(rho_mean_chunks, axis=0),
        np.concatenate(rho_std_chunks, axis=0),
        np.concatenate(ov_mean_chunks, axis=0),
        np.concatenate(ov_std_chunks, axis=0),
        np.concatenate(mae_mean_chunks, axis=0),
    )

# -----------------------------------------------------------------------------
# Publication figures: all main qualitative plots are 1x4
# -----------------------------------------------------------------------------

def plot_curvature_vs_gradient_1x4(img, curvature, gradient, stem, topk=8, alpha=0.55):
    c = _flatten_score(curvature)
    g = _flatten_score(gradient)
    if c.size != g.size:
        raise ValueError("Curvature/gradient token mismatch: %d vs %d" % (c.size, g.size))

    p = pearson_np(c, g)
    r = spearman_np(c, g)
    ov = topk_overlap(c, g, topk)

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    axes[0].imshow(img)
    clean_axis(axes[0], "Input")

    show_heat(axes[1], img, c, "Distilled curvature", alpha=alpha)
    show_heat(axes[2], img, g, "Classification gradient", alpha=alpha)

    draw_topk_comparison(axes[3], img, c, g, topk,
                         a_name="Curvature", b_name="Gradient")
    axes[3].set_title("Top-%d patches\nSpearman=%.3f | O@%d=%.3f" % (topk, r, topk, ov), fontsize=10)

    fig.suptitle("Curvature vs. First-Order Classification Gradient", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)
    return {"pearson": p, "spearman": r, "topk_overlap": ov, "topk": int(topk)}



def _shared_range(a, b, low=1.0, high=99.0):
    """One shared display range for Teacher and Student."""
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    both = np.concatenate([aa, bb], axis=0)
    finite = both[np.isfinite(both)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, low))
    vmax = float(np.percentile(finite, high))
    if vmax <= vmin + 1e-12:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if vmax <= vmin + 1e-12:
        vmax = vmin + 1e-12
    return vmin, vmax


def _percentile_rank01_np(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float64)
    r = _rankdata(x).astype(np.float64)
    r -= np.min(r)
    denom = np.max(r)
    if denom <= 1e-12:
        return np.zeros_like(r)
    return r / denom


def _percentile_rank01_np(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size <= 1:
        return np.zeros_like(x, dtype=np.float64)
    r = _rankdata(x).astype(np.float64)
    r -= np.min(r)
    denom = np.max(r)
    if denom <= 1e-12:
        return np.zeros_like(r)
    return r / denom


def plot_distillation_1x4(
    img,
    teacher,
    student,
    stem,
    topk=8,
    alpha=0.72,
    rho_repeat_mean=float('nan'),
    rho_repeat_std=float('nan'),
    ov_repeat_mean=float('nan'),
    ov_repeat_std=float('nan'),
    mae_repeat_mean=float('nan'),
):
    """Training-time curvature distillation fidelity, 1x4.

    The paper claim here is spatial/rank fidelity, so Teacher and Student are
    displayed as per-image percentile-rank maps on the identical [0,1] scale.
    This does NOT change any metric or training tensor; it only avoids amplitude
    calibration obscuring the spatial ranking. The fourth panel is absolute
    rank error |rank(T)-rank(S)|.
    """
    t = _flatten_score(teacher)
    s = _flatten_score(student)
    if t.size != s.size:
        raise ValueError('Teacher/student token mismatch')

    rho = spearman_np(t, s)
    ov = topk_overlap(t, s, topk)
    mae = float(np.mean(np.abs(t - s)))

    tr = _percentile_rank01_np(t)
    sr = _percentile_rank01_np(s)
    er = np.abs(tr - sr)

    tg = token_grid(tr)
    sg = token_grid(sr)
    eg = token_grid(er)
    h, w = img.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    axes[0].imshow(img)
    clean_axis(axes[0], 'Input')

    axes[1].imshow(img)
    axes[1].imshow(
        resize_grid(tg, h, w), cmap='magma', vmin=0.0, vmax=1.0,
        alpha=alpha, interpolation='bilinear'
    )
    clean_axis(axes[1], 'HVP Teacher\npercentile-rank map')

    axes[2].imshow(img)
    axes[2].imshow(
        resize_grid(sg, h, w), cmap='magma', vmin=0.0, vmax=1.0,
        alpha=alpha, interpolation='bilinear'
    )
    clean_axis(axes[2], 'Student\nsame-forward percentile-rank map')

    axes[3].imshow(
        resize_grid(eg, h, w), cmap='magma', vmin=0.0, vmax=1.0,
        interpolation='bilinear'
    )
    title = 'Absolute rank error\nρ=%.3f | O@%d=%.3f' % (rho, topk, ov)
    if np.isfinite(rho_repeat_mean):
        title += '\nrepeat ρ=%.3f±%.3f' % (rho_repeat_mean, rho_repeat_std)
    clean_axis(axes[3], title)

    fig.suptitle(
        'Training-Time Curvature Distillation Fidelity',
        fontsize=13, fontweight='bold'
    )
    fig.subplots_adjust(left=0.02, right=0.995, bottom=0.07, top=0.84, wspace=0.10)
    save_fig(fig, stem)
    return {
        'pearson_same': pearson_np(t, s),
        'spearman_same': rho,
        'topk_overlap_same': ov,
        'mae_same': mae,
        'spearman_repeat_mean': float(rho_repeat_mean),
        'spearman_repeat_std': float(rho_repeat_std),
        'topk_repeat_mean': float(ov_repeat_mean),
        'topk_repeat_std': float(ov_repeat_std),
        'mae_repeat_mean': float(mae_repeat_mean),
        'topk': int(topk),
    }


def plot_semantic_grounding_1x4(img, category_score, token_semantic, curvature,
                                stem, alpha=0.62):
    """Top-conference qualitative semantic figure: all three token fields are
    rendered with the SAME continuous heat-map convention.

    The previous blue/orange monochrome overlays looked like a global tint and
    obscured spatial peaks.  Here each token field is robust-normalized only for
    display and rendered with ``turbo`` just like the curvature map.
    """
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

    # Same publication heat-map convention for all semantic/localization signals.
    # Semantic maps need slightly stronger overlay than the ordinary heatmap;
    # otherwise blue/orange-like low contrast looks like a uniform image tint.
    sem_alpha = max(float(alpha), 0.65)
    show_heat(axes[1], img, cat, "Category semantic score", alpha=sem_alpha, cmap=HEAT_CMAP)
    show_heat(axes[2], img, sem, "Token-part semantic similarity", alpha=sem_alpha, cmap=HEAT_CMAP)
    show_heat(
        axes[3], img, curv,
        "Distilled curvature\nρ(cat)=%.3f | ρ(local)=%.3f" % (r_cat, r_sem),
        alpha=sem_alpha, cmap=HEAT_CMAP
    )

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


def plot_part_attention_groups_1x4(img, attn_raw, out_dir, topk=8, alpha=0.60):
    """
    Actual part-token attention: attn_raw is [P,N] and is the distribution used
    by part_tokens = attn @ v. We rank queries by concentration and save groups
    of three maps so every qualitative figure stays 1x4 (Input + 3 maps).
    """
    A = np.asarray(attn_raw, dtype=np.float32)
    if A.ndim != 2:
        raise ValueError("attn_raw must be [P,N], got %s" % (A.shape,))
    p, n = A.shape
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError("attn_raw token count is not square")

    score, entropy, peak_ratio = _part_concentration_scores(A)
    order = np.argsort(score)[::-1]
    group_size = 3
    group_id = 0
    for start in range(0, p, group_size):
        chosen = order[start:start + group_size]
        group_id += 1
        fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
        axes[0].imshow(img)
        clean_axis(axes[0], "Input")
        for j in range(3):
            ax = axes[j + 1]
            if j >= len(chosen):
                ax.axis("off")
                continue
            q = int(chosen[j])
            show_heat(
                ax, img, A[q],
                "Part %d attention\npeak×N=%.2f | H=%.2f" % (q + 1, peak_ratio[q], entropy[q]),
                alpha=alpha, cmap=HEAT_CMAP
            )
            # show the user-configurable exact Top-K support with subtle boxes
            draw_topk_boxes(ax, img, A[q], topk, edgecolor="red", linewidth=1.0)
        fig.suptitle("Part-Token Attention Maps (actual attn_raw)", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        save_fig(fig, os.path.join(out_dir, "04_part_attention_group%02d" % group_id))
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



def _topk_part_support(attn_raw, topk):
    """Build an exact-Top-K/query support union and robust part centers."""
    A = np.asarray(attn_raw, dtype=np.float64)
    P, N = A.shape
    side = int(round(math.sqrt(N)))
    if side * side != N:
        raise ValueError("attn_raw token count is not square")

    union = np.zeros(N, dtype=np.float64)
    centers = []

    for q in range(P):
        row = np.maximum(A[q], 0.0)
        idx = topk_indices(row, min(topk, N))
        vals = row[idx].copy()
        vals = vals - vals.min()
        if vals.max() > 1e-12:
            vals = vals / vals.max()
        else:
            vals = np.ones_like(vals)
        union[idx] = np.maximum(union[idx], vals)

        rr = idx // side
        cc = idx % side
        w = np.maximum(row[idx], 0.0)
        if w.sum() <= 1e-12:
            w = np.ones_like(w)
        w = w / w.sum()
        centers.append([float((rr * w).sum()), float((cc * w).sum())])

    return union, np.asarray(centers, dtype=np.float64), side


def plot_part_generation_1x4(
    img,
    curvature,
    token_semantic,
    attn_raw,
    stem,
    topk=8,
    alpha=0.60,
):
    """
    Clean mechanism visualization for the current checkpoint.

    Panel 4 does NOT average eight normalized attention maps (which can wash
    everything into a nearly uniform field). Instead it shows the exact Top-K
    support of each part query and computes each part center only from that
    high-confidence support.
    """
    curv = _flatten_score(curvature)
    sem = _flatten_score(token_semantic)
    A = np.asarray(attn_raw, dtype=np.float32)
    if A.ndim != 2 or A.shape[1] != curv.size or sem.size != curv.size:
        raise ValueError("part generation shape mismatch")

    support, centers, side = _topk_part_support(A, topk=topk)

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    axes[0].imshow(img)
    clean_axis(axes[0], "Input")

    show_heat(
        axes[1],
        img,
        curv,
        "Distilled curvature\nwhere-to-look",
        alpha=alpha,
        cmap=HEAT_CMAP,
    )
    show_heat(
        axes[2],
        img,
        sem,
        "Token-part semantic similarity\nwhat-to-look-for",
        alpha=alpha,
        cmap=HEAT_CMAP,
    )
    show_heat(
        axes[3],
        img,
        support,
        "Part-token support\nTop-%d / query + centers" % int(topk),
        alpha=0.72,
        cmap=DISTILL_CMAP,
    )

    h, w = img.shape[:2]
    ph, pw = h / float(side), w / float(side)
    for q, (rr, cc) in enumerate(centers):
        x = (cc + 0.5) * pw
        y = (rr + 0.5) * ph
        axes[3].scatter(
            [x],
            [y],
            s=30,
            facecolors="white",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        # Small deterministic offsets reduce label collisions without moving centers.
        dx = 3 + (q % 2) * 5
        dy = -3 - (q % 3) * 4
        axes[3].text(
            x + dx,
            y + dy,
            "P%d" % (q + 1),
            color="black",
            fontsize=7,
            fontweight="bold",
            zorder=6,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=0.5),
        )

    fig.suptitle(
        "From Curvature and Semantics to Part-Token Generation",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)
    return {"num_parts": int(A.shape[0]), "topk_per_part": int(topk)}


def plot_cross_stage_1x4(img, stage1_curv, stage2_curv, stem, alpha=0.55):
    """
    Diagnostic ONLY. The current model has no explicit cross-stage consistency
    loss, so low rank correlation is not treated as a method failure.
    """
    c1 = _flatten_score(stage1_curv)
    c2 = _flatten_score(stage2_curv)
    g1 = token_grid(c1)
    g2 = token_grid(c2)
    g2_up = resize_grid(g2, g1.shape[0], g1.shape[1])

    # Compare per-stage percentile ranks rather than independently min-maxed
    # amplitudes; this is a ranking-specialization diagnostic.
    r1 = _rankdata(g1.reshape(-1))
    r1 = ((r1 - 1.0) / max(r1.size - 1, 1)).reshape(g1.shape)
    r2 = _rankdata(g2_up.reshape(-1))
    r2 = ((r2 - 1.0) / max(r2.size - 1, 1)).reshape(g1.shape)
    diff = np.abs(r1 - r2)
    rho = spearman_np(g1.reshape(-1), g2_up.reshape(-1))

    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8))
    axes[0].imshow(img)
    clean_axis(axes[0], "Input")
    show_heat(axes[1], img, c1, "Stage 1 curvature", alpha=alpha)
    show_heat(axes[2], img, c2, "Stage 2 curvature", alpha=alpha)
    axes[3].imshow(
        resize_grid(diff, img.shape[0], img.shape[1]),
        cmap=ERROR_CMAP,
        vmin=0,
        vmax=1,
        interpolation="bilinear",
    )
    clean_axis(
        axes[3],
        "Rank difference\nSpearman=%.3f" % rho,
    )
    fig.suptitle(
        "Cross-Stage Curvature Specialization (diagnostic only)",
        fontsize=13,
        fontweight="bold",
    )
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
    """
    Honest attribute diagnostic.

    ``attr_attn`` can be close to uniform even when weak relative preferences
    exist.  A probability heatmap then looks numerically flat (~1/M).  Because
    log softmax differs from the scaled pre-softmax affinity only by a row-wise
    additive constant, ``log(p) - mean(log(p))`` is a faithful centered
    relative-affinity visualization without inventing a sharper distribution.
    """
    A = np.asarray(attr_attn, dtype=np.float64)
    A = np.squeeze(A)
    if A.ndim != 2:
        raise ValueError("attr_attn must be [P,A]")
    P, M = A.shape
    uniform = 1.0 / float(M)

    probs = np.clip(A, 1e-12, None)
    probs = probs / probs.sum(axis=1, keepdims=True)

    log_aff = np.log(probs)
    log_aff = log_aff - log_aff.mean(axis=1, keepdims=True)
    max_abs = max(float(np.max(np.abs(log_aff))), 1e-6)

    entropy = -(probs * np.log(probs)).sum(axis=1) / math.log(max(M, 2))
    peak_ratio = probs.max(axis=1) / uniform

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.1))
    im = axes[0].imshow(
        log_aff,
        cmap=MATRIX_CMAP,
        vmin=-max_abs,
        vmax=max_abs,
        aspect="auto",
        interpolation="nearest",
    )
    axes[0].set_title("Relative attribute log-affinity\n(centered log probability)")
    axes[0].set_xlabel("Attribute index")
    axes[0].set_ylabel("Part query")
    axes[0].set_yticks(range(P))
    axes[0].set_yticklabels(["P%d" % (i + 1) for i in range(P)])
    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02)

    axes[1].bar(np.arange(P), entropy)
    axes[1].set_ylim(0, 1.02)
    axes[1].axhline(0.98, linestyle="--", linewidth=1.0)
    axes[1].set_title("Normalized attribute entropy\n1.0 = nearly uniform")
    axes[1].set_xlabel("Part query")
    axes[1].set_xticks(range(P))
    axes[1].set_xticklabels(["P%d" % (i + 1) for i in range(P)], rotation=45)

    axes[2].bar(np.arange(P), peak_ratio)
    axes[2].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[2].set_title("Peak / uniform probability\n1.0 = no specialization")
    axes[2].set_xlabel("Part query")
    axes[2].set_xticks(range(P))
    axes[2].set_xticklabels(["P%d" % (i + 1) for i in range(P)], rotation=45)

    fig.suptitle(
        "Part-Attribute Grounding Diagnostic (supplementary)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, stem)

    return {
        "attr_entropy_mean": float(np.mean(entropy)),
        "attr_peak_ratio_mean": float(np.mean(peak_ratio)),
        "attr_log_affinity_absmax": float(max_abs),
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
    ap.add_argument("--teacher-repeats", type=int, default=4,
                    help="Repeated HVP assessment forwards to average. >=4 is recommended for paper figures.")
    ap.add_argument("--hvp-batch-size", type=int, default=2,
                    help="Micro-batch size for second-order HVP evaluation. Use 1 if CUDA memory is tight.")
    ap.add_argument("--gradient-objective", choices=["logit", "ce"], default="logit",
                    help="First-order baseline. logit is recommended; CE can vanish for confident predictions.")
    ap.add_argument("--gradient-class", choices=["pred", "target"], default="pred",
                    help="Class whose logit is differentiated when --gradient-objective logit.")
    ap.add_argument("--gradient-signal", choices=["grad", "gradxinput"], default="grad",
                    help="Token first-order importance signal.")
    ap.add_argument("--save-cross-stage", action="store_true",
                    help="Save cross-stage specialization diagnostic. Disabled by default because the model has no cross-stage consistency objective.")
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
        grad_batch = compute_first_order_importance(
            model,
            samples,
            meta,
            targets,
            args.layer,
            objective=args.gradient_objective,
            class_source=args.gradient_class,
            signal=args.gradient_signal,
        )

        # ---------------------------------------------------------
        # HVP teacher fidelity in the EXACT training distillation space.
        # ---------------------------------------------------------
        teacher_batch = None
        student_train_batch = None
        rho_mean_batch = rho_std_batch = None
        ov_mean_batch = ov_std_batch = None
        mae_mean_batch = None
        if not args.no_teacher:
            try:
                (teacher_batch, student_train_batch,
                 rho_mean_batch, rho_std_batch,
                 ov_mean_batch, ov_std_batch,
                 mae_mean_batch) = collect_teacher_student_fidelity(
                    model, samples, meta, args.layer,
                    repeats=args.teacher_repeats, seed=args.seed + batch_idx * 100,
                    microbatch=args.hvp_batch_size, topk=args.topk
                )
            except Exception as e:
                print("[warning] teacher/student fidelity skipped for batch %d: %r" % (batch_idx, e))
                traceback.print_exc()
                if "out of memory" in str(e).lower():
                    print("[hint] CUDA OOM during HVP: retry with --hvp-batch-size 1 and optionally --teacher-repeats 4")

        # Eval student raw is captured directly from the module.
        selected_inner = inner_eval.get(args.layer)
        if selected_inner is None:
            raise RuntimeError("No internal aux captured for part_gen_%d." % args.layer)

        if not shapes_printed:
            shapes_printed = True
            print("\n[shape validation: selected part_gen_%d]" % args.layer)
            for key in [
                "student_raw", "student_curvature", "student_curvature_reg", "curvature", "curv_weight",
                "hvp_raw", "hvp_curvature", "part_assign", "attn_raw", "part_attn",
                "per_token_sim", "attr_attn", "s_cls"
            ]:
                v = selected_inner.get(key)
                if v is None:
                    print("  %-20s : None" % key)
                elif isinstance(v, torch.Tensor):
                    print("  %-20s : %s" % (key, tuple(v.shape)))
                else:
                    print("  %-20s : %s" % (key, type(v).__name__))
            print("  IMPORTANT: this checkpoint distills mean-normalized predicted curvature vs mean-normalized HVP curvature with Smooth-L1.\n")

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
            # student_curvature: mean-normalized predicted curvature used for distillation
            # curv_weight: positive bounded curvature weight used by classifier path
            # attn_raw: dropout-preceding attention that directly forms part tokens
            student_curvature_eval = batch_item(inner, "student_curvature", i)
            curvature = batch_item(inner, "curv_weight", i)
            if curvature is None:
                curvature = batch_item(inner, "curvature", i)
            gradient = _flatten_score(grad_batch[i])
            s_cls = batch_item(inner, "s_cls", i)
            per_token_sim = batch_item(inner, "per_token_sim", i)
            attn_raw, attn_source = recover_attn_raw(inner, i)
            attr_attn = batch_item(inner, "attr_attn", i, squeeze=False)

            curvature = _flatten_score(curvature)
            student_curvature_eval = _flatten_score(student_curvature_eval)
            s_cls = _flatten_score(s_cls)
            per_token_sim = _flatten_score(per_token_sim)
            attr_attn = np.squeeze(np.asarray(attr_attn, dtype=np.float32)) if attr_attn is not None else None

            n_tokens = curvature.size
            k = min(args.topk, n_tokens)
            fig_status = {}

            # 01: desired full-heatmap style, clean 1x4.
            m = plot_curvature_vs_gradient_1x4(
                img, curvature, gradient,
                os.path.join(image_dir, "01_curvature_vs_gradient"),
                topk=k, alpha=args.heat_alpha
            )
            curv_grad_rows.append({"image": original_name, "layer": args.layer, **m})
            fig_status["01"] = "OK"

            # 02: training-time fidelity only.  Teacher exists only during training,
            # therefore this figure deliberately reproduces the same train-mode path
            # used by the distillation loss.  Inference Student is visualized elsewhere.
            if teacher_batch is not None and student_train_batch is not None:
                dm = plot_distillation_1x4(
                    img,
                    teacher_batch[i],
                    student_train_batch[i],
                    os.path.join(image_dir, "02_distillation_fidelity"),
                    topk=k,
                    alpha=max(args.heat_alpha, 0.68),
                    rho_repeat_mean=float(rho_mean_batch[i]),
                    rho_repeat_std=float(rho_std_batch[i]),
                    ov_repeat_mean=float(ov_mean_batch[i]),
                    ov_repeat_std=float(ov_std_batch[i]),
                    mae_repeat_mean=float(mae_mean_batch[i]),
                )
                dm.update({"image": original_name, "layer": args.layer})
                distill_rows.append(dm)
                fig_status["02"] = "OK"
            else:
                fig_status["02"] = "SKIP: HVP teacher/student unavailable; inspect batch warning above"

            # 03: semantic grounding 1x4.
            if s_cls is not None and per_token_sim is not None:
                sm = plot_semantic_grounding_1x4(
                    img, s_cls, per_token_sim, curvature,
                    os.path.join(image_dir, "03_semantic_grounding"),
                    alpha=args.heat_alpha
                )
                sm.update({"image": original_name, "layer": args.layer})
                semantic_rows.append(sm)
                fig_status["03"] = "OK"
            else:
                fig_status["03"] = "SKIP: s_cls or per_token_sim missing"

            # 04: actual part-token attentions, grouped as 1x4 rows.
            if attn_raw is not None and attn_raw.ndim == 2:
                plot_part_attention_groups_1x4(
                    img, attn_raw, image_dir, topk=k, alpha=args.heat_alpha
                )
                fig_status["04"] = "OK (%s)" % attn_source

                # 05: actual generation path of the uploaded checkpoint.
                # There is NO r17 student_saliency in this architecture; use the
                # real local semantic score plus real attn_raw.
                if per_token_sim is not None:
                    plot_part_generation_1x4(
                        img,
                        curvature,
                        per_token_sim,
                        attn_raw,
                        os.path.join(image_dir, "05_part_generation"),
                        topk=k,
                        alpha=args.heat_alpha,
                    )
                    fig_status["05"] = "OK"
                else:
                    fig_status["05"] = "SKIP: per_token_sim missing"

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
                fig_status["07"] = "OK"
            else:
                msg = "SKIP: %s" % attn_source
                fig_status["04"] = msg
                fig_status["05"] = msg
                fig_status["07"] = msg

            # 06: optional diagnostic only. There is no cross-stage consistency
            # objective in the current architecture, so this is not a main-paper
            # success criterion.
            if args.save_cross_stage:
                if 1 in inner_eval and 2 in inner_eval:
                    c1 = batch_item(inner_eval[1], "curv_weight", i)
                    c2 = batch_item(inner_eval[2], "curv_weight", i)
                    if c1 is not None and c2 is not None:
                        cm = plot_cross_stage_1x4(
                            img,
                            c1,
                            c2,
                            os.path.join(image_dir, "06_cross_stage_diagnostic"),
                            alpha=args.heat_alpha,
                        )
                        cross_rows.append({"image": original_name, **cm})
                        fig_status["06"] = "OK (diagnostic)"
                    else:
                        fig_status["06"] = "SKIP: curv_weight missing in stage 1 or stage 2"
                else:
                    fig_status["06"] = "SKIP: both stages were not captured"
            else:
                fig_status["06"] = "OFF by default (--save-cross-stage to enable)"

            # 08: attribute assignment diagnostic, no misleading 0.03 annotations.
            if attr_attn is not None and attr_attn.ndim == 2:
                am = plot_attribute_assignment(
                    attr_attn, os.path.join(image_dir, "08_attribute_assignment")
                )
                am.update({"image": original_name, "layer": args.layer})
                attr_rows.append(am)
                fig_status["08"] = "OK"
            else:
                fig_status["08"] = "SKIP: attr_attn missing or not [P,A]"

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
            print("       figures: " + " | ".join("%s=%s" % (key, fig_status.get(key, "UNKNOWN"))
                                                     for key in ["01","02","03","04","05","06","07","08"]))
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
        ["image", "layer", "topk",
         "pearson_same", "spearman_same", "topk_overlap_same", "mae_same",
         "spearman_repeat_mean", "spearman_repeat_std",
         "topk_repeat_mean", "topk_repeat_std", "mae_repeat_mean"]
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
        ["image", "layer", "attr_entropy_mean", "attr_peak_ratio_mean",
         "attr_log_affinity_absmax"]
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
    print("Distillation fidelity reproduces the TRAINING-TIME path:")
    print("  student = mean-normalized predicted curvature (same train-mode forward)")
    print("  teacher = mean-normalized HVP curvature (same train-mode forward)")
    print("  display = per-image percentile-rank maps on a shared [0,1] scale")
    print("Teacher repeats = %d" % args.teacher_repeats)
    print("HVP micro-batch size = %d" % args.hvp_batch_size)
    print("First-order baseline = %s / %s / %s" % (
        args.gradient_objective, args.gradient_class, args.gradient_signal
    ))
    print("Part visualization/diversity uses actual dropout-preceding attn_raw used to form part tokens.")
    print("Cross-stage diagnostic is %s." % ("ON" if args.save_cross_stage else "OFF by default"))
    print("======================================")


if __name__ == "__main__":
    main()
