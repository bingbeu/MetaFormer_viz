import argparse
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from visualize import (
    _pm, load_config, unnormalize, token_to_grid, build_model, build_loader
)
import apex.amp

def pearson(a, b):
    """ Pearson correlation coefficient. """
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))

def make_safe_filename(name):
    """ 将原始图片文件名转换为适合 Linux / Windows 的输出文件名。 """
    name = os.path.basename(str(name))
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if not name:
        name = "unknown_image"
    return name

def get_dataset_image_names(dataset):
    """ 尽可能从 Dataset 中获得每张图片的原始文件名。 """
    if hasattr(dataset, "samples"):
        samples = dataset.samples
        names = []
        for item in samples:
            path = item[0] if isinstance(item, (tuple, list)) else item
            names.append(os.path.basename(str(path)))
        if names:
            return names

    if hasattr(dataset, "imgs"):
        imgs = dataset.imgs
        names = []
        for item in imgs:
            path = item[0] if isinstance(item, (tuple, list)) else item
            names.append(os.path.basename(str(path)))
        if names:
            return names

    if hasattr(dataset, "image_paths"):
        paths = dataset.image_paths
        names = [os.path.basename(str(p)) for p in paths]
        if names:
            return names

    if hasattr(dataset, "paths"):
        paths = dataset.paths
        names = [os.path.basename(str(p)) for p in paths]
        if names:
            return names

    if hasattr(dataset, "files"):
        paths = dataset.files
        names = [os.path.basename(str(p)) for p in paths]
        if names:
            return names

    return None

def get_image_names_from_loader(loader):
    """ 从 DataLoader 中寻找 Dataset，并获得图片文件名。 """
    if not hasattr(loader, "dataset"):
        return None
    dataset = loader.dataset

    names = get_dataset_image_names(dataset)
    if names is not None:
        return names

    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
        base_names = get_dataset_image_names(base_dataset)
        if base_names is not None:
            return [base_names[int(idx)] for idx in dataset.indices]

    if hasattr(dataset, "dataset"):
        inner = dataset.dataset
        inner_names = get_dataset_image_names(inner)
        if inner_names is not None:
            return inner_names

    return None

def compute_grad_importance(model, samples, meta, targets, layer, device):
    """ 梯度幅值： || ∂L / ∂x_i ||_2 """
    model.eval()
    model.zero_grad(set_to_none=True)
    out = {}

    def hook(module, inp, outp):
        if isinstance(outp, (tuple, list)):
            outp = outp[0]
        outp.retain_grad()
        out["x"] = outp

    h = getattr(model, f"part_gen_{layer}").input_proj.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            output = model(samples, meta)
            if isinstance(output, torch.Tensor):
                logits = output
            elif isinstance(output, (tuple, list)):
                logits = None
                for item in output:
                    if isinstance(item, torch.Tensor) and (item.ndim == 2 and item.shape[0] == samples.shape[0]):
                        logits = item
                        break
                if logits is None:
                    raise RuntimeError("Cannot find [B,C] logits from model output.")
            else:
                raise RuntimeError(f"Unexpected model output type: {type(output)}")
            loss = F.cross_entropy(logits, targets.to(device))
            loss.backward()
    finally:
        h.remove()

    if "x" not in out:
        raise RuntimeError("Forward hook did not capture part_gen input_proj.")
    if out["x"].grad is None:
        raise RuntimeError("Captured tensor has no gradient.")
    return out["x"].grad.norm(p=2, dim=-1).detach().cpu()

def plot_curv_vs_grad(img, curv, grad, save_path, tag):
    """ 可视化 Curvature vs Gradient """
    curv = np.squeeze(curv)
    grad = np.squeeze(grad)
    curv01 = (curv - curv.min()) / (curv.max() - curv.min() + 1e-8)
    grad01 = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(img)
    axes[0].set_title("Input", fontsize=12)
    axes[0].axis("off")

    for ax, m, title in [(axes[1], curv01, "Curvature (2nd-order)"), (axes[2], grad01, "Gradient (1st-order)")]:
        hm = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        hm = F.interpolate(hm, size=(img.shape[0], img.shape[1]), mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(img, alpha=0.5)
        ax.imshow(hm, cmap="turbo", alpha=0.55)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    flat_c = curv.flatten()
    flat_g = grad.flatten()
    r = pearson(torch.from_numpy(flat_c), torch.from_numpy(flat_g))
    axes[3].scatter(flat_g, flat_c, s=6, c=flat_c - flat_g, cmap="coolwarm", alpha=0.6)
    axes[3].set_xlabel("Gradient magnitude")
    axes[3].set_ylabel("Curvature")
    axes[3].set_title(f"Per-token scatter (Pearson r={r:.3f})")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return r

def select_part_locations(part_assign, topk=8, min_distance=0):
    """ 从每个 Part Query 中选择 Top-1。默认不人为修改模型选择结果。 """
    if isinstance(part_assign, torch.Tensor):
        part_assign = part_assign.detach().cpu().numpy()
    part_assign = np.asarray(part_assign)
    if part_assign.ndim != 2:
        raise ValueError(f"part_assign must have shape (N,P), got {part_assign.shape}")
    N, P = part_assign.shape
    topk = min(topk, P)

    dummy = torch.from_numpy(part_assign[:, 0:1]).float()
    grid = token_to_grid(dummy).detach().cpu().numpy().squeeze()
    H_grid, W_grid = grid.shape

    part_min = part_assign[:, :topk].min(axis=0, keepdims=True)
    part_max = part_assign[:, :topk].max(axis=0, keepdims=True)
    part_norm = (part_assign[:, :topk] - part_min) / (part_max - part_min + 1e-8)

    selected = []
    for p in range(topk):
        token_id = int(np.argmax(part_norm[:, p]))
        row, col = np.unravel_index(token_id, (H_grid, W_grid))
        score = float(part_norm[token_id, p])
        selected.append((p, token_id, int(row), int(col), score))

    return selected, H_grid, W_grid

def plot_part_attention_grid(img, part_assign, save_path, topk=8, min_distance=0):
    """ 8 个 Part 子图，每个子图：原图 + 一个红色 patch 方框。 """
    if isinstance(part_assign, torch.Tensor):
        part_assign_np = part_assign.detach().cpu().numpy()
    else:
        part_assign_np = np.asarray(part_assign)

    selected, H_grid, W_grid = select_part_locations(part_assign_np, topk=topk, min_distance=min_distance)

    H_img, W_img = img.shape[:2]
    patch_h = H_img / float(H_grid)
    patch_w = W_img / float(W_grid)

    cols = 4
    rows = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, 6.8))
    axes = axes.ravel()

    for idx, (part_id, token_id, row, col, score) in enumerate(selected):
        ax = axes[idx]
        ax.imshow(img)
        x = col * patch_w
        y = row * patch_h
        rect = Rectangle((x, y), patch_w, patch_h, fill=False, linewidth=2.5, edgecolor="red")
        ax.add_patch(rect)
        ax.set_title(f"Part {part_id + 1}", fontsize=12, fontweight="bold")
        ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    fig.suptitle("Part-wise Top-1 Discriminative Evidence Localization", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n[Part Top-1]")
    for part_id, token_id, row, col, score in selected:
        print(f" Part {part_id + 1}: token={token_id}, grid=({row},{col}), score={score:.6f}")

def get_student_teacher(model, samples, meta, layer, device):
    """ 获得 student predicted curvature 和 teacher HVP curvature """
    was_training = model.training
    model.train()
    try:
        with torch.enable_grad():
            output = model(samples.to(device), meta, return_aux=True)
    finally:
        model.train(was_training)

    aux = None
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, dict):
                aux = item
                break
    if aux is None:
        raise RuntimeError("Cannot find aux dictionary.")

    student_key = f"curvature_{layer}"
    if student_key not in aux:
        raise KeyError(f"Missing {student_key}. Available keys: {list(aux.keys())}")
    student = aux[student_key].squeeze(-1).detach().cpu()

    teacher_key = f"hvp_curvature_{layer}"
    if teacher_key not in aux:
        raise KeyError(f"Missing {teacher_key}. Available keys: {list(aux.keys())}")
    teacher = aux[teacher_key].squeeze(-1).detach().cpu()

    return student, teacher

def plot_distill_scatter(student, teacher, save_path):
    """ Student vs teacher HVP curvature """
    s = (student - student.mean()) / (student.std() + 1e-8)
    t = (teacher - teacher.mean()) / (teacher.std() + 1e-8)
    r = pearson(s, t)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, s, s=6, alpha=0.5)
    lim = [min(t.min(), s.min()), max(t.max(), s.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="y=x")
    ax.set_xlabel("Teacher HVP curvature (standardized)")
    ax.set_ylabel("Student predicted curvature (standardized)")
    ax.set_title(f"Distillation fidelity (Pearson r={r:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="./figs_novelty")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--layer", type=int, default=2, choices=[1, 2])
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--part-topk", type=int, default=8)
    ap.add_argument("--part-min-distance", type=int, default=0, help="Minimum spatial distance. 0 means no spatial constraint.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.cfg)
    cfg.defrost()
    cfg.EVAL_MODE = True
    cfg.MODEL.assess = True
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    if not hasattr(apex.amp, "_amp_state"):
        apex.amp._amp_state = type("DummyState", (), {})()
    if not hasattr(apex.amp._amp_state, "handle") or apex.amp._amp_state.handle is None:
        apex.amp._amp_state.handle = type("DummyHandle", (), {"_is_active": False})()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, pickle_module=_pm)
    sd = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    sd = {(k.replace("module.", "", 1) if k.startswith("module.") else k): v for k, v in sd.items()}

    hw = sd.get("head.weight")
    if hw is not None:
        cfg.defrost()
        cfg.MODEL.NUM_CLASSES = hw.shape[0]
        cfg.freeze()

    model = build_model(cfg)
    model.load_state_dict(sd, strict=False)
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
    if image_names is not None:
        print(f"[info] Found {len(image_names)} image filenames.")
        print("[info] Example:")
        for name in image_names[:5]:
            print(f" {name}")
    else:
        print("[warning] Could not automatically find image filenames from Dataset.")
        print("[warning] Visualization will fallback to img{i} naming.")

    L = args.layer
    n_imgs = 0
    dataset_index = 0
    rs = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        if cfg.DATA.ADD_META:
            samples, targets, meta = batch
            meta = [m.float() for m in meta]
            meta = torch.stack(meta, dim=0).to(device)
        else:
            samples, targets = batch
            meta = None

        samples = samples.to(device)
        targets = targets.to(device)
        B = samples.shape[0]

        with torch.no_grad():
            output = model(samples, meta, return_aux=True)

        aux = None
        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, dict):
                    aux = item
                    break
        if aux is None:
            raise RuntimeError("Cannot find aux dictionary.")

        grad = compute_grad_importance(model, samples, meta, targets, L, device)
        curv = aux[f"curvature_{L}"]

        part_key = f"part_assign_{L}"
        if part_key not in aux:
            raise KeyError(f"Missing {part_key}. Available keys:\n{list(aux.keys())}")
        part_assign = aux[part_key]

        for i in range(B):
            if n_imgs >= args.num_images:
                break
            img = unnormalize(samples[i].detach().cpu())

            if image_names is not None and dataset_index < len(image_names):
                original_name = image_names[dataset_index]
                safe_name = make_safe_filename(original_name)
            else:
                original_name = f"img{n_imgs}"
                safe_name = f"img{n_imgs}"

            print("\n------------------------------------------------")
            print(f"[image {n_imgs}]")
            print(f"dataset index : {dataset_index}")
            print(f"original file : {original_name}")
            print(f"output name : {safe_name}")

            c = token_to_grid(curv[i]).detach().cpu().numpy().squeeze()
            g = token_to_grid(grad[i]).detach().cpu().numpy().squeeze()

            curv_path = os.path.join(args.out, f"curv_vs_grad_{safe_name}.png")
            r = plot_curv_vs_grad(img, c, g, curv_path, n_imgs)
            rs.append(r)

            part_path = os.path.join(args.out, f"part_attention_{safe_name}.png")
            plot_part_attention_grid(img, part_assign[i], part_path, topk=args.part_topk, min_distance=args.part_min_distance)
            print(f"[save] {curv_path}")
            print(f"[save] {part_path}")

            dataset_index += 1
            n_imgs += 1
            if n_imgs >= args.num_images:
                break

    print("\n[info] 计算蒸馏保真度 （train 模式 + HVP）...")
    s_all = []
    t_all = []
    n_distill = 0
    max_distill = 128
    for batch in loader:
        if cfg.DATA.ADD_META:
            samples, targets, meta = batch
            meta = [m.float() for m in meta]
            meta = torch.stack(meta, dim=0).to(device)
        else:
            samples, targets = batch
            meta = None
        samples = samples.to(device)
        s, t = get_student_teacher(model, samples, meta, L, device)
        s_all.append(s)
        t_all.append(t)
        n_distill += samples.shape[0]
        if n_distill >= max_distill:
            break

    s_cat = torch.cat(s_all, dim=0) if s_all else torch.tensor([])
    t_cat = torch.cat(t_all, dim=0) if t_all else torch.tensor([])

    if s_cat.numel() == 0:
        print("[warning] No distillation samples.")
    else:
        dr = plot_distill_scatter(s_cat, t_cat, os.path.join(args.out, "distill_scatter.png"))
        print(f"[save] distill_scatter (r={dr:.3f})")

    mean_r = np.mean(rs) if rs else float("nan")
    print("\n================================================")
    print("完成！")
    print(f"曲率-梯度相关系数均值 = {mean_r:.3f}")
    print("\n文件命名方式：")
    print(" curv_vs_grad_<原图文件名>.png")
    print(" part_attention_<原图文件名>.png")
    print("\n例如：")
    print(" 原图：")
    print(" Black_Footed_Albatross_0046_18.jpg")
    print(" 输出：")
    print(" curv_vs_grad_Black_Footed_Albatross_0046_18.png")
    print(" part_attention_Black_Footed_Albatross_0046_18.png")
    print("\n输出目录：")
    print(os.path.abspath(args.out))
    print("================================================")

if __name__ == "__main__":
    main()