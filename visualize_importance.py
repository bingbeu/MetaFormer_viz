"""
用法：
  python visualize_importance.py --cfg /raid/viz/MetaFormer_vis/output/MetaFG_meta_2/cub-200-vis/config.json --ckpt /raid/viz/MetaFormer_vis/output/MetaFG_meta_2/cub-200-vis/best.pth  --out ./figs_novelty-2 --num-images 20 --layer 2
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualize import _pm, load_config, unnormalize, token_to_grid, build_model, build_loader
from matplotlib.patches import Rectangle
import apex.amp

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-8))


def compute_grad_importance(model, samples, meta, targets, layer, device):
    """梯度幅值 ||∂L/∂x_i||_2，与曲率同定义在 token 特征上。"""
    model.eval()
    out = {}
    def hook(module, inp, outp):
        outp.retain_grad(); out["x"] = outp
    h = getattr(model, f"part_gen_{layer}").input_proj.register_forward_hook(hook)
    with torch.enable_grad():
        logits = model(samples, meta, return_aux=False)[0]   # 取第一个返回值为分类输出
        F.cross_entropy(logits, targets.to(device)).backward()
    h.remove()
    return out["x"].grad.norm(p=2, dim=-1).detach().cpu()   # (B,N)


# ---------------------------------------------------------------------------
# Fig2: 曲率 vs 梯度 —— 证明是互补信号
# ---------------------------------------------------------------------------
def plot_curv_vs_grad(img, curv, grad, save_path, tag):
    """左上：原图；中左：曲率图；中右：梯度图；最右：散点(带相关系数)。"""
    curv = np.squeeze(curv); grad = np.squeeze(grad)
    curv01 = (curv - curv.min()) / (curv.max() - curv.min() + 1e-8)
    grad01 = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

    # 修改为 1行4列，调整图像比例
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5)) 
    axes[0].imshow(img); axes[0].set_title("input"); axes[0].axis("off")

    for ax, m, t in [(axes[1], curv01, "curvature (2nd-order)"),
                     (axes[2], grad01, "gradient (1st-order)")]:
        hm = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        hm = F.interpolate(hm, size=(img.shape[0], img.shape[1]),
                           mode="bilinear", align_corners=False)[0, 0].numpy()
        ax.imshow(img, alpha=0.5); ax.imshow(hm, cmap="turbo", alpha=0.55)
        ax.set_title(t); ax.axis("off")

    # 最右侧散点图
    flat_c, flat_g = curv.flatten(), grad.flatten()
    r = pearson(torch.from_numpy(flat_c), torch.from_numpy(flat_g))
    axes[3].scatter(flat_g, flat_c, s=6, c=flat_c - flat_g, cmap="coolwarm", alpha=0.6)
    axes[3].set_xlabel("gradient magnitude"); axes[3].set_ylabel("curvature")
    axes[3].set_title(f"per-token scatter (Pearson r={r:.3f})")
    
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return r


def plot_part_attention_grid(img, part_assign, save_path, topk=8):
    """part_assign (N,P) -> (H,W,P)。展示 top-k 个 part query 各自关注的区域。"""
    P = part_assign.shape[-1]
    topk = min(topk, P)
    maps = []
    for p in range(topk):
        m = token_to_grid(part_assign[:, p:p+1]).numpy().squeeze()   # (H,W)
        maps.append(m)
    cols = 4
    rows = int(np.ceil(topk / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()
    
    for p, (ax, m) in enumerate(zip(axes, maps)):
        hm = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
        hm = F.interpolate(hm, size=(img.shape[0], img.shape[1]),
                           mode="bilinear", align_corners=False)[0, 0].numpy()
        
        # 新修改：只凸显最重要的区域（例如 75% 分位数以上的区域，也可改为 hm.max() * 0.6）
        threshold = np.percentile(hm, 90) 
        # 使用 np.ma.masked_where 将低值区域设为透明
        masked_hm = np.ma.masked_where(hm < threshold, hm)
        
        # 先画清晰的原图，再叠加热力图
        ax.imshow(img) 
        ax.imshow(masked_hm, cmap="turbo", alpha=0.8) # 透明度可适当调高，只凸显重点区域
        ax.set_title(f"part {p}"); ax.axis("off")
        
    for ax in axes[topk:]:
        ax.axis("off")
    fig.suptitle("each part query attends to a distinct region (structured parts)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
# ---------------------------------------------------------------------------
# Fig4: 学生 vs 老师 HVP —— 证明蒸馏保真
# ---------------------------------------------------------------------------
def get_student_teacher(model, samples, meta, layer, device):
    was_training = model.training
    model.train()
    with torch.enable_grad():
        cls, aux, w1, w2 = model(samples.to(device), meta, return_aux=True)
    model.train(was_training)

    student = aux[f"curvature_{layer}"].squeeze(-1).detach().cpu()
    teacher = aux[f"hvp_curvature_{layer}"].squeeze(-1).detach().cpu()
    if teacher is None:
        raise RuntimeError("hvp_curvature 为 None……")

    return student, teacher


def plot_distill_scatter(student, teacher, save_path):
    # 将 student 和 teacher 分别标准化（均值为0，方差为1），消除尺度影响
    s = (student - student.mean()) / (student.std() + 1e-8)
    t = (teacher - teacher.mean()) / (teacher.std() + 1e-8)
    r = pearson(s, t)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, s, s=6, alpha=0.5)
    lim = [min(t.min(), s.min()), max(t.max(), s.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="y=x")
    ax.set_xlabel("teacher HVP curvature (standardized)")
    ax.set_ylabel("student predicted curvature (standardized)")
    ax.set_title(f"distillation fidelity (Pearson r={r:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return r


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="./figs_novelty")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--layer", type=int, default=2, choices=[1, 2])
    ap.add_argument("--max-batches", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.cfg)
    cfg.defrost(); cfg.EVAL_MODE = True; cfg.MODEL.assess = True; cfg.freeze()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 修补 apex.amp 的 handle（避免 AMP 未初始化时报错）
    if not hasattr(apex.amp, '_amp_state'):
        apex.amp._amp_state = type('DummyState', (), {})()
    if not hasattr(apex.amp._amp_state, 'handle') or apex.amp._amp_state.handle is None:
        apex.amp._amp_state.handle = type('DummyHandle', (), {'_is_active': False})()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, pickle_module=_pm)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    hw = sd.get("head.weight")
    if hw is not None:
        cfg.defrost(); cfg.MODEL.NUM_CLASSES = hw.shape[0]; cfg.freeze()

    model = build_model(cfg)
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    # ====== 修改点1：强制开启 assess ======
    model.assess = True
    # ====================================

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29503")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo")

    _, _, _, loader, _ = build_loader(cfg)
    L = args.layer
    n_imgs = 0
    rs = []

    for batch in loader:
        if cfg.DATA.ADD_META:
            samples, targets, meta = batch
            meta = [m.float() for m in meta]
            meta = torch.stack(meta, dim=0).to(device)
        else:
            samples, targets = batch
            meta = None
        samples = samples.to(device); targets = targets.to(device)
        B = samples.shape[0]

        with torch.no_grad():
            _, aux, attn1, attn2 = model(samples, meta, return_aux=True)
        grad = compute_grad_importance(model, samples, meta, targets, L, device)

        curv = aux[f"curvature_{L}"]
        part_assign = aux[f"part_assign_{L}"]

        for i in range(B):
            if n_imgs >= args.num_images:
                break
            img = unnormalize(samples[i].cpu())
            c = token_to_grid(curv[i]).numpy().squeeze()
            g = token_to_grid(grad[i]).numpy().squeeze()

            r = plot_curv_vs_grad(img, c, g,
                                  os.path.join(args.out, f"curv_vs_grad_img{n_imgs}.png"), n_imgs)
            rs.append(r)

            plot_part_attention_grid(img, part_assign[i].detach().cpu(),
                                     os.path.join(args.out, f"part_attention_img{n_imgs}.png"))

            print(f"[save] img{n_imgs}: curv_vs_grad (r={r:.3f}) + part_attention")
            n_imgs += 1
        if n_imgs >= args.num_images:
            break

    # ====== 修改点2：蒸馏保真度——使用全部样本，而非仅前2张 ======
    print("\n[info] 计算蒸馏保真度（train 模式 + HVP，使用全部 batch 样本）...")
    s_all, t_all = [], []
    n_distill = 0
    max_distill = 128   # 至少收集128张图，保证统计稳定
    for batch in loader:
        if cfg.DATA.ADD_META:
            samples, targets, meta = batch
            meta = [m.float() for m in meta]
            meta = torch.stack(meta, dim=0).to(device)
        else:
            samples, targets = batch
            meta = None
        samples = samples.to(device)
        s, t = get_student_teacher(model, samples, meta, L, device)   # 使用整个 batch
        s_all.append(s)
        t_all.append(t)
        n_distill += samples.shape[0]
        if n_distill >= max_distill:
            break
    # =============================================================

    s_cat = torch.cat(s_all, dim=0) if s_all else torch.tensor([])
    t_cat = torch.cat(t_all, dim=0) if t_all else torch.tensor([])
    if s_cat.numel() == 0:
        print("[警告] 未收集到蒸馏样本，跳过 Fig4")
    else:
        dr = plot_distill_scatter(s_cat, t_cat, os.path.join(args.out, "distill_scatter.png"))
        print(f"[save] distill_scatter (r={dr:.3f})")

    print(f"\n完成！曲率-梯度相关系数均值 = {np.mean(rs):.3f}")
    print("判读：r 越低说明曲率与梯度越互补（二阶≠一阶）；若 r 接近 1 说明曲率没提供新信息，要警惕。")
    print(f"输出目录：{os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()