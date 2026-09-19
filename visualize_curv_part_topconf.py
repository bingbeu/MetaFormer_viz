import argparse
import os
import re
import math
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.ndimage as ndimage
from visualize import (_pm, load_config, unnormalize, token_to_grid, build_model, build_loader)

# ---------- utilities ----------
def pearson(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / (denom + 1e-12))

def spearman(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return pearson(ra, rb)

def norm01(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo + 1e-12)

def entropy01(x):
    x = np.asarray(x, dtype=float).reshape(-1)
    x = np.maximum(x, 0.0)
    s = x.sum()
    if s <= 1e-12:
        return float("nan")
    p = x / s
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))

def make_safe_filename(name):
    name = os.path.basename(str(name))
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name if name else "unknown_image"

def get_dataset_image_names(dataset):
    for attr in ("samples", "imgs", "image_paths", "paths", "files"):
        if hasattr(dataset, attr):
            vals = getattr(dataset, attr)
            names = []
            for item in vals:
                path = item[0] if isinstance(item, (tuple, list)) else item
                names.append(os.path.basename(str(path)))
            if names:
                return names
    return None

def get_image_names_from_loader(loader):
    if not hasattr(loader, "dataset"):
        return None
    dataset = loader.dataset
    names = get_dataset_image_names(dataset)
    if names is not None:
        return names
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_names = get_dataset_image_names(dataset.dataset)
        if base_names is not None:
            return [base_names[int(i)] for i in dataset.indices]
    if hasattr(dataset, "dataset"):
        names = get_dataset_image_names(dataset.dataset)
        if names is not None:
            return names
    return None

def batch_item(x, index, batch_size):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        if len(x) == batch_size:
            return x[index]
        if len(x) > 0:
            x = x[0]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.ndim > 0 and x.shape[0] == batch_size:
        return x[index]
    return x

def vectorize_token(x, index, batch_size, n_hint=None):
    x = batch_item(x, index, batch_size)
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().float().cpu().squeeze()
    if x.ndim == 0:
        return x.reshape(1).numpy()
    if x.ndim == 1:
        return x.numpy()
    if n_hint is not None:
        matching = [d for d, s in enumerate(x.shape) if s == n_hint]
        if matching:
            token_dim = matching[-1]
            y = x.movedim(token_dim, 0).reshape(n_hint, -1)
            y = torch.linalg.vector_norm(y, ord=2, dim=1)
            return y.numpy()
    y = x.reshape(x.shape[0], -1)
    y = torch.linalg.vector_norm(y, ord=2, dim=1)
    return y.numpy()

def infer_grid(token_vector):
    n = int(np.asarray(token_vector).size)
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"Token count {n} is not a square.")
    return side, side

def upsample_map(m, image_shape):
    m = np.asarray(m, dtype=float)
    t = torch.from_numpy(m).float()[None, None]
    up = F.interpolate(t, size=image_shape, mode="bilinear", align_corners=False)[0, 0]
    return up.numpy()

def overlay_map(ax, image, token_map, title, cmap="turbo"):
    token_map = norm01(token_map)
    ax.imshow(image)
    ax.imshow(upsample_map(token_map, image.shape[:2]), cmap=cmap, alpha=0.52, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

def save_figure(fig, path, dpi=300):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ---------- gradient importance ----------
def first_logits_from_output(output, batch_size):
    if isinstance(output, torch.Tensor) and output.ndim == 2 and output.shape[0] == batch_size:
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 2 and item.shape[0] == batch_size:
                return item
    raise RuntimeError("Cannot find [B,C] classification logits.")

def compute_grad_importance(model, samples, meta, targets, layer, device):
    model.eval()
    model.zero_grad(set_to_none=True)
    captured = {}
    module = getattr(model, f"part_gen_{layer}").input_proj
    def hook(_module, _inp, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        output.retain_grad()
        captured["x"] = output
    handle = module.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            output = model(samples, meta)
            logits = first_logits_from_output(output, samples.shape[0])
            loss = F.cross_entropy(logits, targets.to(device))
            loss.backward()
    finally:
        handle.remove()
    if "x" not in captured:
        raise RuntimeError("Gradient hook did not capture part_gen input_proj.")
    if captured["x"].grad is None:
        raise RuntimeError("Captured tensor has no gradient.")
    return captured["x"].grad.norm(p=2, dim=-1).detach().cpu()

# ---------- attention extraction ----------
def extract_attention_maps_single(weight_tensor, extra_token_num):
    if not torch.is_tensor(weight_tensor):
        weight_tensor = torch.as_tensor(weight_tensor)
    w = weight_tensor.detach().cpu().float()
    if w.ndim == 5:
        w = w[0, 0]
    elif w.ndim == 4:
        w = w[0]
    if w.ndim != 3:
        raise ValueError(f"Attention must become [heads,S,S], got {tuple(w.shape)}")
    heads, seq_len, _ = w.shape
    if seq_len <= extra_token_num:
        raise ValueError(f"seq_len={seq_len} <= EXTRA_TOKEN_NUM={extra_token_num}")
    cls_to_image = w[:, 0, extra_token_num:]
    n = cls_to_image.shape[-1]
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"Attention image-token count {n} is not square.")
    return cls_to_image.reshape(heads, side, side).numpy()

def get_attention_for_sample(layer_weights, sample_index, batch_size, attn_layer_idx):
    if layer_weights is None:
        return None
    if isinstance(layer_weights, (list, tuple)):
        if len(layer_weights) == 0:
            return None
        w = layer_weights[attn_layer_idx] if attn_layer_idx < len(layer_weights) else layer_weights[0]
    else:
        w = layer_weights
    if not torch.is_tensor(w):
        w = torch.as_tensor(w)
    if w.ndim == 5:
        if w.shape[0] == batch_size:
            return w[sample_index, attn_layer_idx]
        return w[attn_layer_idx, sample_index]
    if w.ndim == 4:
        if w.shape[0] == batch_size:
            return w[sample_index]
        if attn_layer_idx < w.shape[0]:
            return w[attn_layer_idx]
        return w[0]
    if w.ndim == 3:
        return w
    raise ValueError(f"Unsupported attention tensor shape: {tuple(w.shape)}")

# ---------- semantic attribute attention ----------
def extract_attribute_maps(attr_attn, sample_index, batch_size, n_tokens):
    x = batch_item(attr_attn, sample_index, batch_size)
    if x is None:
        return None
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().float().cpu().squeeze()
    if x.ndim == 1:
        if x.numel() == n_tokens:
            return x.reshape(1, n_tokens).numpy()
        return None
    if x.ndim != 2:
        return None
    if x.shape[0] == n_tokens:
        return x.T.numpy()
    if x.shape[1] == n_tokens:
        return x.numpy()
    return None

# ---------- figures ----------
def plot_curvature_vs_gradient(image, curvature, gradient, save_path):
    c, g = norm01(curvature), norm01(gradient)
    r, rho = pearson(curvature, gradient), spearman(curvature, gradient)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(image)
    axes[0].set_title("Input", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    overlay_map(axes[1], image, c, "Curvature (2nd-order)", cmap="Reds")
    overlay_map(axes[2], image, g, "Gradient (1st-order)", cmap="Blues")
    axes[3].scatter(g.reshape(-1), c.reshape(-1), s=8, alpha=0.45)
    axes[3].set_xlabel("Gradient magnitude")
    axes[3].set_ylabel("Curvature")
    axes[3].set_title(f"Per-token relation\nPearson r={r:.3f}, Spearman ρ={rho:.3f}", fontsize=10)
    save_figure(fig, save_path)
    return r, rho

def plot_teacher_student(image, teacher, student, save_path):
    r, rho = pearson(teacher, student), spearman(teacher, student)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    overlay_map(axes[0], image, teacher, "HVP Teacher", cmap="Reds")
    overlay_map(axes[1], image, student, "First-order Student", cmap="Greens")
    t, s = norm01(teacher).reshape(-1), norm01(student).reshape(-1)
    axes[2].scatter(t, s, s=8, alpha=0.45)
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[2].set_xlabel("Teacher curvature")
    axes[2].set_ylabel("Student prediction")
    axes[2].set_title(f"Distillation fidelity\nPearson r={r:.3f}, Spearman ρ={rho:.3f}", fontsize=10)
    fig.suptitle("Online Curvature Distillation", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)
    return r, rho

def plot_semantic_grounding(image, attr_maps, save_path, max_attributes=3):
    if attr_maps is None or attr_maps.shape[0] == 0:
        return False
    k = min(max_attributes, attr_maps.shape[0])
    fig, axes = plt.subplots(1, k + 1, figsize=(3.3 * (k + 1), 3.8))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    axes[0].set_title("Input", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    for j in range(k):
        overlay_map(axes[j+1], image, attr_maps[j], f"Attribute {j+1}", cmap="Oranges")
    fig.suptitle("Attribute-conditioned Semantic Grounding", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)
    return True

def plot_curvature_vs_semantics(image, curvature, semantic_similarity, save_path):
    r, rho = pearson(curvature, semantic_similarity), spearman(curvature, semantic_similarity)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    overlay_map(axes[0], image, curvature, "Curvature", cmap="Reds")
    overlay_map(axes[1], image, semantic_similarity, "Token semantic similarity", cmap="Oranges")
    c, s = norm01(curvature).reshape(-1), norm01(semantic_similarity).reshape(-1)
    axes[2].scatter(s, c, s=8, alpha=0.45)
    axes[2].set_xlabel("Semantic similarity")
    axes[2].set_ylabel("Curvature")
    axes[2].set_title(f"Semantic relation\nPearson r={r:.3f}, Spearman ρ={rho:.3f}", fontsize=10)
    fig.suptitle("Curvature on the Category–Attribute Alignment Space", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)
    return r, rho

def plot_localization_policy(image, teacher, student, weight, gate, save_path):
    maps, titles, cmaps = [teacher, student, weight], ["HVP teacher", "Student importance", "Curv-weight"], ["Reds", "Greens", "Blues"]
    if gate is not None:
        maps.append(gate); titles.append("Gate"); cmaps.append("Purples")
    fig, axes = plt.subplots(1, len(maps), figsize=(3.3 * len(maps), 3.8))
    axes = np.atleast_1d(axes)
    for ax, m, title, cmap in zip(axes, maps, titles, cmaps):
        overlay_map(ax, image, m, title, cmap=cmap)
    fig.suptitle("Curvature → Explicit Differentiable Localization Policy", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)

def select_part_locations(part_assign, topk=5):
    pa = np.asarray(part_assign)
    if pa.ndim != 2:
        raise ValueError(f"part_assign must be [N,P], got {pa.shape}")
    n_tokens, n_parts = pa.shape
    topk = min(topk, n_parts)
    selected = []
    for p in range(topk):
        score = norm01(pa[:, p])
        token_id = int(np.argmax(score))
        h, w = infer_grid(score)
        row, col = np.unravel_index(token_id, (h, w))
        selected.append((p, token_id, int(row), int(col), float(score[token_id])))
    return selected, h, w

def plot_part_localization(image, part_assign, save_path, topk=5):
    selected, h, w = select_part_locations(part_assign, topk)
    fig, axes = plt.subplots(1, len(selected), figsize=(3.0 * len(selected), 3.5))
    axes = np.atleast_1d(axes)
    patch_h, patch_w = image.shape[0] / float(h), image.shape[1] / float(w)
    for ax, (p, token_id, row, col, score) in zip(axes, selected):
        ax.imshow(image)
        rect = Rectangle((col * patch_w, row * patch_h), patch_w, patch_h,
                         fill=False, linewidth=2.2, edgecolor="#2D9B4E")
        ax.add_patch(rect)
        ax.set_title(f"Part P{p+1}", fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Curvature-guided Part Localization", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)

def compute_curvature_part_alignment(curvature, part_assign, top_fraction=0.10):
    c = np.asarray(curvature).reshape(-1)
    pa = np.asarray(part_assign)
    selected, _, _ = select_part_locations(pa, topk=pa.shape[1])
    k = max(1, int(round(len(c) * top_fraction)))
    top_curv = set(np.argsort(c)[-k:].tolist())
    order = np.argsort(-c)
    rank_map = np.empty_like(order)
    rank_map[order] = np.arange(len(order))
    hits, ranks = [], []
    for item in selected:
        _, token_id, _, _, _ = item
        hits.append(float(token_id in top_curv))
        ranks.append(float(rank_map[token_id]) / max(1, len(c) - 1))
    return float(np.mean(hits)), float(np.mean(ranks))

def plot_curvature_part_alignment(curvature, part_assign, save_path):
    pa = np.asarray(part_assign)
    c = np.asarray(curvature).reshape(-1)
    selected, _, _ = select_part_locations(pa, topk=pa.shape[1])
    part_ids, curv_values = [], []
    for item in selected:
        p, token_id, _, _, _ = item
        part_ids.append(p+1)
        curv_values.append(norm01(c)[token_id])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(part_ids, curv_values)
    ax.set_xlabel("Selected part")
    ax.set_ylabel("Normalized curvature at selected part token")
    ax.set_xticks(part_ids)
    ax.set_title("Does curvature concentrate at selected part tokens?", fontsize=11, fontweight="bold")
    save_figure(fig, save_path)

def plot_attention_maps(image, attention_maps, save_path, max_heads=4):
    if attention_maps is None:
        return
    k = min(max_heads, attention_maps.shape[0])
    fig, axes = plt.subplots(1, k + 1, figsize=(3.0 * (k + 1), 3.5))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    axes[0].set_title("Input", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    for j in range(k):
        overlay_map(axes[j+1], image, attention_maps[j], f"Head {j+1}", cmap="Blues")
    fig.suptitle("Transformer Attention", fontsize=13, fontweight="bold")
    save_figure(fig, save_path)

def plot_weight_distribution(weight, gate, save_path):
    w = np.asarray(weight).reshape(-1)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(w, bins=30, alpha=0.75)
    ax.set_xlabel("Curvature weight")
    ax.set_ylabel("Token count")
    ax.set_title("Token-importance Weight Distribution", fontsize=11, fontweight="bold")
    if gate is not None:
        g = np.asarray(gate).reshape(-1)
        ax.text(0.98, 0.95, f"Gate mean = {np.mean(g):.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    save_figure(fig, save_path)

def save_csv(rows, path):
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

def get_student_teacher(model, samples, meta, layer, device):
    old_training = model.training
    model.train()
    try:
        with torch.enable_grad():
            output = model(samples.to(device), meta, return_aux=True)
    finally:
        model.train(old_training)
    aux = None
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, dict):
                aux = item
                break
    if aux is None:
        raise RuntimeError("Cannot find auxiliary dictionary in model output.")
    student_key, teacher_key = f"curvature_{layer}", f"hvp_curvature_{layer}"
    if student_key not in aux or teacher_key not in aux:
        raise KeyError(f"Missing teacher/student keys. Available: {list(aux.keys())}")
    return aux[student_key].detach().cpu(), aux[teacher_key].detach().cpu()

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="./figs_novelty")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--layer", type=int, default=2, choices=[1, 2])
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--part-topk", type=int, default=5)
    parser.add_argument("--attn-layer-idx", type=int, default=0)
    parser.add_argument("--max-distill", type=int, default=128)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = load_config(args.cfg)
    cfg.defrost()
    cfg.EVAL_MODE = True
    cfg.MODEL.assess = True
    cfg.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, pickle_module=_pm)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state_dict.items()}
    if "head.weight" in state_dict:
        cfg.defrost()
        cfg.MODEL.NUM_CLASSES = state_dict["head.weight"].shape[0]
        cfg.freeze()
    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warning] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warning] unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    model.assess = True

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo")

    _, _, _, loader, _ = build_loader(cfg)
    image_names = get_image_names_from_loader(loader)
    if image_names is not None:
        print(f"[info] Found {len(image_names)} image filenames.")
    else:
        print("[warning] Could not recover dataset filenames.")
    layer = args.layer
    extra_token_num = getattr(cfg.MODEL, "EXTRA_TOKEN_NUM", 17)
    print(f"[info] EXTRA_TOKEN_NUM = {extra_token_num}")

    metric_rows = []
    n_images, dataset_index = 0, 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches or n_images >= args.num_images:
            break
        if cfg.DATA.ADD_META:
            samples, targets, meta = batch
            meta = torch.stack([m.float() for m in meta], dim=0).to(device)
        else:
            samples, targets = batch
            meta = None
        samples, targets = samples.to(device), targets.to(device)
        batch_size = samples.shape[0]

        with torch.no_grad():
            output = model(samples, meta, return_aux=True)
        if isinstance(output, (tuple, list)) and len(output) == 4:
            cls_out, aux, layer_weights1, layer_weights2 = output
        else:
            aux = None
            if isinstance(output, (tuple, list)):
                for item in output:
                    if isinstance(item, dict):
                        aux = item
                        break
            if aux is None:
                raise RuntimeError("Cannot find aux dictionary.")
            layer_weights1 = layer_weights2 = None

        curvature = aux[f"curvature_{layer}"]
        print(f"[DEBUG] curvature type: {type(curvature)}, shape: {curvature.shape if hasattr(curvature, 'shape') else 'no shape'}")
        hvp_curvature = aux[f"hvp_curvature_{layer}"]
        part_assign = aux.get(f"part_assign_{layer}")
        attr_attn = aux.get(f"attr_attn_{layer}")
        curv_weight = aux.get(f"curv_weight_{layer}")
        gate_status = aux.get(f"gate_status_{layer}")
        per_token_sim = aux.get(f"per_token_sim_{layer}")

        gradient = compute_grad_importance(model, samples, meta, targets, layer, device)

        for i in range(batch_size):
            if n_images >= args.num_images:
                break
            image = unnormalize(samples[i].detach().cpu())
            original_name = image_names[dataset_index] if image_names and dataset_index < len(image_names) else f"img{n_images}"
            safe_name = make_safe_filename(original_name)
            image_dir = os.path.join(args.out, safe_name)
            os.makedirs(image_dir, exist_ok=True)
            plt.imsave(os.path.join(image_dir, f"original_{safe_name}.jpg"), image)
            print(f"\n----------------------------------------")
            print(f"[image {n_images}] dataset_index={dataset_index}")
            print(f"[file] {original_name}")

            c_vec = vectorize_token(curvature, i, batch_size)
            print(f"[DEBUG] c_vec: {c_vec}")
            g_vec = vectorize_token(gradient, i, batch_size, n_hint=c_vec.size)
            t_vec = vectorize_token(hvp_curvature, i, batch_size, n_hint=c_vec.size)
            h_grid, w_grid = infer_grid(c_vec)
            c_map, g_map, t_map = c_vec.reshape(h_grid, w_grid), g_vec.reshape(h_grid, w_grid), t_vec.reshape(h_grid, w_grid)

            curv_grad_path = os.path.join(image_dir, f"01_curvature_vs_gradient_{safe_name}.png")
            r_cg, rho_cg = plot_curvature_vs_gradient(image, c_map, g_map, curv_grad_path)

            distill_path = os.path.join(image_dir, f"02_teacher_student_{safe_name}.png")
            r_ts, rho_ts = plot_teacher_student(image, t_map, c_map, distill_path)

            attr_maps = extract_attribute_maps(attr_attn, i, batch_size, c_vec.size)
            attr_path = os.path.join(image_dir, f"03_semantic_grounding_{safe_name}.png")
            attr_ok = plot_semantic_grounding(image, attr_maps, attr_path) if attr_maps is not None else False

            r_cs, rho_cs = float("nan"), float("nan")
            if per_token_sim is not None:
                sim_vec = vectorize_token(per_token_sim, i, batch_size, n_hint=c_vec.size)
                if sim_vec is not None and sim_vec.size == c_vec.size:
                    sim_map = sim_vec.reshape(h_grid, w_grid)
                    semantic_path = os.path.join(image_dir, f"04_curvature_vs_semantics_{safe_name}.png")
                    r_cs, rho_cs = plot_curvature_vs_semantics(image, c_map, sim_map, semantic_path)

            weight_vec, gate_vec = None, None
            if curv_weight is not None:
                weight_vec = vectorize_token(curv_weight, i, batch_size, n_hint=c_vec.size)
            if gate_status is not None:
                gate_vec = vectorize_token(gate_status, i, batch_size, n_hint=c_vec.size)
            if weight_vec is not None and weight_vec.size == c_vec.size:
                weight_map = weight_vec.reshape(h_grid, w_grid)
                gate_map = gate_vec.reshape(h_grid, w_grid) if gate_vec is not None and gate_vec.size == c_vec.size else None
                policy_path = os.path.join(image_dir, f"05_localization_policy_{safe_name}.png")
                plot_localization_policy(image, t_map, c_map, weight_map, gate_map, policy_path)
                stability_path = os.path.join(image_dir, f"06_weight_distribution_{safe_name}.png")
                plot_weight_distribution(weight_vec, gate_vec, stability_path)

            curv_part_hit, curv_part_rank = float("nan"), float("nan")
            if part_assign is not None:
                pa_i = batch_item(part_assign, i, batch_size)
                if pa_i is not None:
                    pa_i = pa_i.detach().cpu().numpy() if torch.is_tensor(pa_i) else np.asarray(pa_i)
                    part_path = os.path.join(image_dir, f"07_part_localization_{safe_name}.png")
                    plot_part_localization(image, pa_i, part_path, topk=args.part_topk)
                    curv_part_hit, curv_part_rank = compute_curvature_part_alignment(c_vec, pa_i)
                    mechanism_path = os.path.join(image_dir, f"08_curvature_part_alignment_{safe_name}.png")
                    plot_curvature_part_alignment(c_vec, pa_i, mechanism_path)

            attention_saved = False
            if layer_weights1 is not None:
                try:
                    w_sample = get_attention_for_sample(layer_weights1, i, batch_size, args.attn_layer_idx)
                    attention_maps = extract_attention_maps_single(w_sample, extra_token_num)
                    attention_path = os.path.join(image_dir, f"09_transformer_attention_{safe_name}.png")
                    plot_attention_maps(image, attention_maps, attention_path)
                    attention_saved = True
                except Exception as exc:
                    print(f"[warning] attention visualization skipped: {exc}")

            row = {
                "image": original_name,
                "layer": layer,
                "curv_grad_pearson": r_cg,
                "curv_grad_spearman": rho_cg,
                "teacher_student_pearson": r_ts,
                "teacher_student_spearman": rho_ts,
                "curv_semantic_pearson": r_cs,
                "curv_semantic_spearman": rho_cs,
                "curv_part_top10pct_hit": curv_part_hit,
                "curv_part_rank_norm": curv_part_rank,
                "attribute_attention_available": int(attr_ok),
                "transformer_attention_available": int(attention_saved),
            }
            if weight_vec is not None:
                row.update({
                    "curv_weight_mean": float(np.mean(weight_vec)),
                    "curv_weight_std": float(np.std(weight_vec)),
                    "curv_weight_min": float(np.min(weight_vec)),
                    "curv_weight_max": float(np.max(weight_vec)),
                    "curv_weight_entropy": entropy01(weight_vec),
                })
            if gate_vec is not None:
                row.update({"gate_mean": float(np.mean(gate_vec)), "gate_std": float(np.std(gate_vec))})
            metric_rows.append(row)
            print(f"[metric] curvature-gradient r={r_cg:.4f}")
            print(f"[metric] teacher-student r={r_ts:.4f}")
            if not np.isnan(r_cs):
                print(f"[metric] curvature-semantic r={r_cs:.4f}")
            if not np.isnan(curv_part_hit):
                print(f"[metric] part tokens inside top-10% curvature = {curv_part_hit:.4f}")
            dataset_index += 1
            n_images += 1

    # global distillation
    print("\n[info] Computing global teacher-student fidelity...")
    student_all, teacher_all = [], []
    n_distill = 0
    for batch in loader:
        if n_distill >= args.max_distill:
            break
        if cfg.DATA.ADD_META:
            samples, targets, meta = batch
            meta = torch.stack([m.float() for m in meta], dim=0).to(device)
        else:
            samples, targets = batch
            meta = None
        samples = samples.to(device)
        student, teacher = get_student_teacher(model, samples, meta, layer, device)
        student, teacher = student.squeeze(), teacher.squeeze()
        if student.ndim == 1: student = student.unsqueeze(0)
        if teacher.ndim == 1: teacher = teacher.unsqueeze(0)
        if student.ndim > 2: student = student.reshape(student.shape[0], -1)
        if teacher.ndim > 2: teacher = teacher.reshape(teacher.shape[0], -1)
        n_common = min(student.shape[-1], teacher.shape[-1])
        student_all.append(student[:, :n_common])
        teacher_all.append(teacher[:, :n_common])
        n_distill += samples.shape[0]
    if student_all:
        student_cat = torch.cat(student_all, dim=0).numpy()
        teacher_cat = torch.cat(teacher_all, dim=0).numpy()
        s_flat, t_flat = student_cat.reshape(-1), teacher_cat.reshape(-1)
        global_r, global_rho = pearson(s_flat, t_flat), spearman(s_flat, t_flat)
        global_path = os.path.join(args.out, "10_distillation_global.png")
        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        ax.scatter(norm01(t_flat), norm01(s_flat), s=6, alpha=0.35)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("HVP teacher curvature")
        ax.set_ylabel("Student predicted importance")
        ax.set_title(f"Global distillation fidelity\nPearson r={global_r:.3f}, Spearman ρ={global_rho:.3f}", fontsize=11, fontweight="bold")
        save_figure(fig, global_path)
        print(f"[distill] Pearson={global_r:.4f}, Spearman={global_rho:.4f}")
    else:
        print("[warning] No samples collected for global distillation plot.")

    # save CSVs
    csv_path = os.path.join(args.out, "per_image_mechanism_metrics.csv")
    save_csv(metric_rows, csv_path)
    if metric_rows:
        summary = {}
        metric_names = [
            "curv_grad_pearson", "curv_grad_spearman",
            "teacher_student_pearson", "teacher_student_spearman",
            "curv_semantic_pearson", "curv_semantic_spearman",
            "curv_part_top10pct_hit", "curv_part_rank_norm",
            "curv_weight_mean", "curv_weight_std", "curv_weight_min", "curv_weight_max", "curv_weight_entropy",
            "gate_mean", "gate_std"
        ]
        for key in metric_names:
            values = [float(row[key]) for row in metric_rows if key in row and row[key] is not None and np.isfinite(float(row[key]))]
            if values:
                summary[key] = float(np.mean(values))
        summary_path = os.path.join(args.out, "mechanism_summary.csv")
        save_csv([summary], summary_path)

    print("\n================================================")
    print("Curv-Part visualization finished.")
    print(f"Images processed: {n_images}")
    print(f"Output directory: {args.out}")
    print("\nImportant mechanism figures:")
    print("  01_curvature_vs_gradient_*")
    print("  02_teacher_student_*")
    print("  03_semantic_grounding_*")
    print("  04_curvature_vs_semantics_*")
    print("  05_localization_policy_*")
    print("  06_weight_distribution_*")
    print("  07_part_localization_*")
    print("  08_curvature_part_alignment_*")
    print("  09_transformer_attention_*")
    print("  10_distillation_global.png")
    print("\nQuantitative mechanism tables:")
    print("  per_image_mechanism_metrics.csv")
    print("  mechanism_summary.csv")
    print("================================================")

if __name__ == "__main__":
    main()