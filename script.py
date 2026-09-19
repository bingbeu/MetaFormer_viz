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
import scipy.ndimage as ndimage

# ---------- 公用函数 ----------
def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))

def make_safe_filename(name):
    name = os.path.basename(str(name))
    name = os.path.splitext(name)[0]
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    if not name:
        name = "unknown_image"
    return name

def get_dataset_image_names(dataset):
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

# ---------- 梯度计算 ----------
def compute_grad_importance(model, samples, meta, targets, layer, device):
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

# ---------- 注意力提取（修正版） ----------
def extract_attention_maps(weight_tensor, batch_idx=0, layer_idx=0, extra_token_num=17, up_scale=16):
    """
    从注意力权重中提取指定样本、指定层的注意力图。
    支持输入形状：
      - [B, heads, seq, seq]
      - [layers, B, heads, seq, seq]
      - [B, layers, heads, seq, seq]  （自动识别，以batch为第一维优先）
    返回: [heads, H, W] (numpy)
    """
    if isinstance(weight_tensor, torch.Tensor):
        w = weight_tensor.detach().cpu().numpy()
    else:
        w = np.asarray(weight_tensor)

    ndim = w.ndim
    # 根据维度自动推断
    if ndim == 4:
        # [B, heads, seq, seq]
        w_sample = w[batch_idx]  # [heads, seq, seq]
    elif ndim == 5:
        # 先判断第一维是否等于可能的batch数（通常batch较小）
        # 简单启发：如果第一维 == 第二维的某个倍数？更可靠：检查第一维是否等于batch大小（由外部传入？）
        # 此处我们假设第一维为batch，第二维为层数（常见情况），否则假设第一维为层数，第二维为batch
        # 通过形状大小粗略判断：如果第一维<=4且第二维较大，则认为第一维是层数；否则第一维是batch。
        # 但更稳健：让用户通过参数指定，这里我们自动尝试：优先尝试 batch, layers
        if w.shape[0] > w.shape[1]:  # 第一维更大，可能是layers？但batch也可能大。不准确。
            # 我们简单地尝试：如果第一维等于传入的batch_idx范围，就当作batch
            # 但batch_idx可能为0，无法判断。改为打印信息并假设 [batch, layers, heads, seq, seq]
            # 更好的做法：在调用时明确指定维度顺序。
            # 这里提供两种常见顺序的自动适配：
            # 如果 w.shape[1] 与典型的层数（如2）相近，而 w.shape[0] 是batch，则按 [B, L, H, S, S]
            # 否则按 [L, B, H, S, S]
            # 简单处理：默认第一维为batch，第二维为层数。
            pass
        # 默认：第一维 = batch, 第二维 = layers
        w_sample = w[batch_idx, layer_idx, :, :, :]  # [heads, seq, seq]
    else:
        raise ValueError(f"Unsupported weight ndim: {ndim}, shape: {w.shape}")

    # 现在 w_sample 应为 [heads, seq, seq]
    n_heads, seq_len, _ = w_sample.shape
    # 取 CLS token (索引0) 对图像 token 的注意力
    img_att = w_sample[:, 0, extra_token_num:]   # [heads, N_patches]
    N_patches = img_att.shape[-1]
    Hf = Wf = int(np.sqrt(N_patches))
    if Hf * Wf != N_patches:
        raise ValueError(f"N_patches={N_patches} is not a perfect square. Check extra_token_num.")
    img_map = img_att.reshape(n_heads, Hf, Wf)   # [heads, Hf, Wf]
    # 上采样
    img_att_up = ndimage.zoom(img_map, (1, up_scale, up_scale), order=0)
    return img_att_up

# ---------- 网格图绘制 ----------
def show_grid_images(imgs, rows, cols, titles=None, scale=3, cmap='rainbow'):
    figsize = (cols * scale, rows * scale)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], cmap=cmap)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.axes.set_title(titles[i])
    return fig, axes

def show_individual(imgs, layer_id, save_root, prefix=""):
    converge_dir = os.path.join(save_root, f'{prefix}_converge')
    os.makedirs(converge_dir, exist_ok=True)
    sum_img = imgs.mean(0)
    avg_path = os.path.join(converge_dir, f'layer_{layer_id:02d}_avg.jpg')
    plt.imsave(avg_path, sum_img)
    head_dir = os.path.join(save_root, f'{prefix}_layer_{layer_id+1:02d}_attention_maps')
    os.makedirs(head_dir, exist_ok=True)
    for i in range(imgs.shape[0]):
        save_path = os.path.join(head_dir, f'head_{i:02d}.jpg')
        plt.imsave(save_path, imgs[i])
    print(f"  Saved layer {layer_id} attention maps to {save_root}")

# ---------- 曲率 vs 梯度 ----------
def plot_curv_vs_grad(img, curv, grad, save_path, tag):
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

# ---------- Part 选择与可视化 ----------
def select_part_locations(part_assign, topk=8, min_distance=0):
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
    fig, axes = plt.subplots(rows, cols, figsize=(12.8, 7.5), gridspec_kw={'hspace': 0.4, 'wspace': 0.05})
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

# ---------- 蒸馏保真度 ----------
def get_student_teacher(model, samples, meta, layer, device):
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
    teacher_key = f"hvp_curvature_{layer}"
    if student_key not in aux or teacher_key not in aux:
        raise KeyError(f"Missing keys. Available: {list(aux.keys())}")
    student = aux[student_key].squeeze(-1).detach().cpu()
    teacher = aux[teacher_key].squeeze(-1).detach().cpu()
    return student, teacher

def plot_distill_scatter(student, teacher, save_path):
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

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="./figs_novelty")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--layer", type=int, default=2, choices=[1, 2])
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument("--part-topk", type=int, default=8)
    ap.add_argument("--part-min-distance", type=int, default=0)
    ap.add_argument("--attn-layer-idx", type=int, default=0, help="Layer index inside layer_weights (if multi-layer)")
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
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo")

    _, _, _, loader, _ = build_loader(cfg)

    image_names = get_image_names_from_loader(loader)
    if image_names is not None:
        print(f"[info] Found {len(image_names)} image filenames.")
        for name in image_names[:5]:
            print(f" {name}")
    else:
        print("[warning] Could not automatically find image filenames from Dataset.")

    L = args.layer
    n_imgs = 0
    dataset_index = 0
    rs = []

    extra_token_num = getattr(cfg.MODEL, 'EXTRA_TOKEN_NUM', 17)
    print(f"[info] Using EXTRA_TOKEN_NUM = {extra_token_num}")

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

        # 解包输出
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
            layer_weights1 = None
            print("[warning] Model output is not a 4-element tuple, skipping attention visualization.")

        if aux is None:
            raise RuntimeError("aux dictionary is None.")
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
            print(f"[image {n_imgs}] dataset index {dataset_index} file {original_name}")

            # 曲率 vs 梯度
            c = token_to_grid(curv[i]).detach().cpu().numpy().squeeze()
            g = token_to_grid(grad[i]).detach().cpu().numpy().squeeze()
            curv_path = os.path.join(args.out, f"curv_vs_grad_{safe_name}.png")
            r = plot_curv_vs_grad(img, c, g, curv_path, n_imgs)
            rs.append(r)

            # Part Attention
            part_path = os.path.join(args.out, f"part_attention_{safe_name}.png")
            plot_part_attention_grid(img, part_assign[i], part_path, topk=args.part_topk, min_distance=args.part_min_distance)
            print(f"[save] {curv_path}")
            print(f"[save] {part_path}")

            # ========== 注意力可视化（修正：处理 list 类型） ==========
            if layer_weights1 is not None:
                # ---- 处理 layer_weights1 可能是 list 的情况 ----
                if isinstance(layer_weights1, (list, tuple)):
                    # 假设列表按层顺序，args.layer 为1或2，映射索引 0 或 1
                    layer_idx = args.layer - 1
                    if layer_idx < len(layer_weights1):
                        w = layer_weights1[layer_idx]
                        print(f"[info] Using layer_weights1[{layer_idx}] for layer {args.layer}")
                    else:
                        print(f"[warning] layer_weights1 has length {len(layer_weights1)}, using first.")
                        w = layer_weights1[0]
                else:
                    w = layer_weights1

                # 确保 w 是 Tensor
                if not isinstance(w, torch.Tensor):
                    if isinstance(w, np.ndarray):
                        w = torch.from_numpy(w)
                    else:
                        raise TypeError(f"Unsupported type for w: {type(w)}")

                print(f"[debug] w.shape = {w.shape}")
                # ---- 从 w 中提取第 i 个样本的注意力权重 ----
                w_shape = w.shape
                if w_shape[0] == B:  # 第一维是 batch
                    if w.ndim == 5:
                        # [B, layers, heads, seq, seq]
                        w_sample = w[i, args.attn_layer_idx, :, :, :]  # [heads, seq, seq]
                    elif w.ndim == 4:
                        w_sample = w[i]  # [heads, seq, seq]
                    else:
                        raise ValueError(f"Unsupported ndim: {w.ndim}")
                else:
                    # 可能第一维是层数，第二维是 batch
                    if w.ndim == 5:
                        w_sample = w[args.attn_layer_idx, i, :, :, :]
                    else:
                        raise ValueError(f"Cannot infer dimension order from shape {w_shape}")

                # 现在 w_sample 是 [heads, seq, seq]
                # 调用 extract_attention_maps，传入单个样本（加 batch 维度）
                attn_maps = extract_attention_maps(
                    w_sample.unsqueeze(0),  # 加 batch 维度 -> [1, heads, seq, seq]
                    batch_idx=0,
                    layer_idx=0,
                    extra_token_num=extra_token_num,
                    up_scale=16
                )  # 返回 [heads, H, W]

                # 创建子目录
                img_dir = os.path.join(args.out, safe_name)
                os.makedirs(img_dir, exist_ok=True)

                # 保存原图
                orig_path = os.path.join(img_dir, 'original.jpg')
                plt.imsave(orig_path, img)
                print(f"[save] {orig_path}")

                # 网格图
                grid_fig, _ = show_grid_images(attn_maps, rows=2, cols=4)
                grid_path = os.path.join(img_dir, 'attention_grid.png')
                grid_fig.savefig(grid_path, dpi=200, bbox_inches='tight')
                plt.close(grid_fig)
                print(f"[save] {grid_path}")

                # 单头和平均图
                show_individual(attn_maps, layer_id=args.attn_layer_idx, save_root=img_dir, prefix="")
                print(f"[save] attention maps to {img_dir}")
            else:
                print("[info] layer_weights1 not available, skip attention visualization.")

            dataset_index += 1
            n_imgs += 1
            if n_imgs >= args.num_images:
                break

    # 蒸馏保真度
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
    print("\n文件保存结构：")
    print(" <输出目录>/curv_vs_grad_<原图>.png")
    print(" <输出目录>/part_attention_<原图>.png")
    print(" <输出目录>/<原图>/original.jpg          # 原图")
    print(" <输出目录>/<原图>/attention_grid.png    # 注意力网格")
    print(" <输出目录>/<原图>/converge/             # 平均图")
    print(" <输出目录>/<原图>/layer_*_attention_maps/ # 各head")
    print("================================================")

if __name__ == "__main__":
    main()