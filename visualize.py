"""
visualize.py —— 用 best.pth 生成论文用的可视化图
=================================================
生成三类图（以 part_gen_2 为例，可用 --layer 1 看 part_gen_1）：
  ① curvature 热力图      —— 模型"看哪里"（曲率高的 token 更亮）
  ② part_assign 部件分配图 —— 每个 token 被分给哪个部件（离散色）
  ③ attr_attn 属性-部件矩阵 —— 每个部件关注哪些属性 token
  ④ curv_weight 直方/Top-K —— 曲率权重的分布

前置条件（重要）：
  1. 已在 forward_features / forward 里应用"返回完整 aux"的 3 处改动
     （即 model(images, meta, return_aux=True) 返回 (logits, aux_full)，
      aux_full 含 curvature_1/2、part_assign_1/2、attr_attn_1/2、curv_weight_1/2 ...）。
     如果还没改，先改好再跑本脚本。
  2. 在模型仓库根目录运行（需要能 import 到你的 build_model / build_loader）。

用法：
  python visualize.py --cfg /raid/viz/MetaFormer_viz/output/MetaFG_2/stanfordcars-5-cont/config.json --ckpt /raid/viz/MetaFormer_viz/output/MetaFG_2/stanfordcars-5-cont/best.pth \
         --out ./figs_cars --num-images 6 --layer 2
"""

import argparse
import os
import pickle
import types

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# 0) 安全反序列化（checkpoint 里可能带 yacs config 对象，缺 yacs 时跳过）
# ---------------------------------------------------------------------------
class _Dummy(dict):
    def __init__(self, *a, **k): dict.__init__(self)
    def __getattr__(self, name): return _Dummy()
    def __call__(self, *a, **k): return _Dummy()
    def __getitem__(self, k): return _Dummy()
    def __setitem__(self, k, v): pass
    def __setattr__(self, k, v): pass

class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            return _Dummy

_pm = types.ModuleType("safepickle")
_pm.Unpickler = _SafeUnpickler


# ---------------------------------------------------------------------------
# 1) 适配区：改成你仓库的实际导入（这是唯一需要你核对的地方）
# ---------------------------------------------------------------------------
try:
    from main import build_model, build_loader       # 你的 main.py（有 __main__ 保护时可 import）
except Exception as e:                                # 若不行，改成你的实际路径，如：
    try:
        from models import build_model                # from models.model_builder import build_model
        from data import build_loader                 # from data.build import build_loader
    except Exception as e2:
        raise ImportError(
            "无法导入 build_model / build_loader。请打开 visualize.py 的『适配区』，"
            f"改成你仓库里的实际导入路径。\n  main import err: {e}\n  fallback err: {e2}"
        )

try:
    from yacs.config import CfgNode
    _HAS_YACS = True
except ImportError:
    _HAS_YACS = False


# ---------------------------------------------------------------------------
# 2) 小工具
# ---------------------------------------------------------------------------
def load_config(cfg_path):
    """config.json 是 yacs dump 出的 YAML 格式内容，但扩展名是 .json，
    yacs 的 merge_from_file 只认 .yaml/.yml/.py，所以用 yaml 直接解析。"""
    if not _HAS_YACS:
        raise RuntimeError("需要 yacs（pip install yacs）")
    try:
        import yaml
    except ImportError:
        raise RuntimeError("需要 pyyaml（pip install pyyaml）")
    with open(cfg_path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    cfg = CfgNode(d)
    return cfg


def unnormalize(img_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """把归一化后的 (C,H,W) 张量转成可显示的 0-255 uint8 图。
    如果数据增强用的 mean/std 不同，改这里的默认值。"""
    img = img_tensor.detach().cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def token_to_grid(token_map, h=None, w=None):
    """把 token 级张量 reshape 成 (B, H, W)，返回 CPU 张量。
    兼容输入：
      (B, N)     —— 批量 token 图
      (B, N, 1)  —— 带尾维（如 curvature）
      (N,)       —— 单样本 1D（如 part_assign.argmax(-1)）
      (N, 1)     —— 单样本带尾维
    N 必须能开平方（part_gen 的输入是方形特征图）。"""
    t = token_map.detach().cpu()
    # 去掉所有长度为 1 的尾维：(B,N,1)->(B,N)，(N,1)->(N,)
    while t.dim() >= 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    if t.dim() == 1:                       # 单样本 (N,) -> (1, N)
        t = t.unsqueeze(0)
    if t.dim() != 2:
        raise ValueError(f"token_to_grid 无法处理 shape={tuple(token_map.shape)}")
    B, N = t.shape
    if h is None or w is None:
        side = int(round(N ** 0.5))
        if side * side != N:
            raise ValueError(f"N={N} 不是平方数，请手动指定 --h/--w")
        h = w = side
    return t.reshape(B, h, w)


# ---------------------------------------------------------------------------
# 3) 绘图函数
# ---------------------------------------------------------------------------
def plot_curvature(img, curv_grid, save_path, title="curvature"):
    """原图 + 曲率热力图叠加。"""
    curv_grid = np.squeeze(curv_grid)          # 去掉可能的 batch 维 (1,H,W) -> (H,W)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("input")
    axes[0].axis("off")

    h, w = curv_grid.shape
    # 上采样到原图尺寸
    from torch.nn.functional import interpolate
    hm = torch.from_numpy(curv_grid).float().unsqueeze(0).unsqueeze(0)
    hm = interpolate(hm, size=(img.shape[0], img.shape[1]), mode="bilinear", align_corners=False)
    hm = hm[0, 0].numpy()

    im = axes[1].imshow(img, alpha=0.6)
    im2 = axes[1].imshow(hm, cmap="turbo", alpha=0.55)
    axes[1].set_title(title)
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_part_assign(img, assign_grid, save_path, num_parts, title="part assignment"):
    """部件分配图：每个 token 颜色 = 它被分到的部件。"""
    assign_grid = np.squeeze(assign_grid)      # 去掉可能的 batch 维 (1,H,W) -> (H,W)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img)
    axes[0].set_title("input")
    axes[0].axis("off")

    cmap = plt.get_cmap("tab20", num_parts)
    axes[1].imshow(img, alpha=0.45)
    im = axes[1].imshow(assign_grid, cmap=cmap, vmin=-0.5, vmax=num_parts - 0.5, alpha=0.65)
    axes[1].set_title(f"{title} (P={num_parts})")
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], ticks=np.arange(num_parts), fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([f"p{i}" for i in range(num_parts)])
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_attr_attn(attr_attn, save_path, num_parts):
    """属性-部件注意力矩阵 (P, A)。A==1 时跳过并提示。"""
    P, A = attr_attn.shape
    if A == 1:
        print(f"[skip] attr_attn 的 A={A}（meta 只有 1 个属性 token），属性矩阵无意义")
        return
    fig, ax = plt.subplots(figsize=(max(4, A * 0.5), max(3, P * 0.4)))
    im = ax.imshow(attr_attn, cmap="viridis", aspect="auto")
    ax.set_xlabel("attribute token idx")
    ax.set_ylabel("part idx")
    ax.set_title(f"attribute-part attention (P={P}, A={A})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_curv_weight(curv_weight, save_path, topk=10):
    """单个样本的 curv_weight 分布（Top-K 柱状图）。"""
    w = curv_weight.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(w, bins=50)
    axes[0].set_title(f"curv_weight hist (mean={w.mean():.3f}, max={w.max():.2f})")
    axes[0].set_xlabel("weight")

    idx = np.argsort(w)[::-1][:topk]
    axes[1].bar(np.arange(topk), w[idx])
    axes[1].set_xticks(np.arange(topk))
    axes[1].set_xticklabels([str(i) for i in idx], rotation=90)
    axes[1].set_title("top-k token weights")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="训练时的 config.json 路径")
    ap.add_argument("--ckpt", required=True, help="best.pth 路径（用 cub-200-V3/best.pth）")
    ap.add_argument("--out", default="./figs", help="输出目录")
    ap.add_argument("--num-images", type=int, default=6)
    ap.add_argument("--layer", type=int, default=2, choices=[1, 2],
                    help="可视化 part_gen_1 还是 part_gen_2")
    ap.add_argument("--max-batches", type=int, default=20,
                    help="在验证集里最多扫多少个 batch 找可用样本")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.cfg)
    cfg.defrost()
    cfg.EVAL_MODE = True                 # 只用验证集
    cfg.freeze()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    # ---- checkpoint 先加载（用它的 head 形状推断类别数）----
    try:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False, pickle_module=_pm)
    except TypeError:                       # 老版本 torch 没有 weights_only 参数
        ckpt = torch.load(args.ckpt, map_location="cpu", pickle_module=_pm)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    # 若 checkpoint 是 DDP 存的，去掉 module. 前缀
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}

    # 训练时 NUM_CLASSES 被覆盖成了数据集类别数（CUB=200），而 config.json 里是 1000。
    # 从 checkpoint 的 head.weight 推断并覆盖，避免分类头尺寸不匹配。
    head_w = sd.get("head.weight")
    if head_w is not None:
        num_classes = head_w.shape[0]
        cfg.defrost()
        cfg.MODEL.NUM_CLASSES = num_classes
        cfg.freeze()
        print(f"[info] override NUM_CLASSES -> {num_classes}（从 checkpoint head 推断）")

    # ---- 模型 ----
    model = build_model(cfg)
    model.load_state_dict(sd, strict=False)   # 若报 missing key 再改 strict=True
    model = model.to(device).eval()
    print(f"[info] checkpoint loaded: epoch={ckpt.get('epoch')} max_acc={ckpt.get('max_accuracy')}")

    # ---- 数据（复用你的验证集 loader，yield (samples, targets, meta)）----
    # build_loader 内部会调 dist.get_rank()；独立运行 visualize.py 时
    # 没有分布式进程组，这里初始化一个单进程 group（不影响单卡推理）。
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="gloo")
        print("[info] initialized single-process dist group (gloo, rank 0 / world 1)")
    _, _, _, data_loader_val, _ = build_loader(cfg)   # 与 main() 中 build_loader 返回顺序一致
    L = args.layer
    n_imgs = 0
    attr_attn_sum = None
    n_batch = 0

    with torch.no_grad():
        for batch in data_loader_val:
            if cfg.DATA.ADD_META:
                samples, targets, meta = batch
                meta = [m.float() for m in meta]
                meta = torch.stack(meta, dim=0).to(device)
            else:
                samples, targets = batch
                meta = None
            samples = samples.to(device)

            try:
                logits, aux_full = model(samples, meta, return_aux=True)
            except Exception as e:
                raise RuntimeError(
                    "model(..., return_aux=True) 没有返回完整 aux。请先应用『返回完整 aux』"
                    f"的 3 处改动（见 visualize.py 文件头说明）。\n  err: {e}"
                )

            B = samples.shape[0]

            # 属性-部件矩阵（跨 batch 累加均值）——放在图片数量 break 之前，
            # 否则 num-images 一旦满足就会提前退出，属性矩阵永远画不出来。
            a = aux_full[f"attr_attn_{L}"]
            if attr_attn_sum is None:
                attr_attn_sum = a.sum(0).detach().cpu().float()
            else:
                attr_attn_sum += a.sum(0).detach().cpu().float()
            n_batch += B

            for i in range(B):
                if n_imgs >= args.num_images:
                    break
                img = unnormalize(samples[i].cpu())
                curv = aux_full[f"curvature_{L}"][i]                 # (N,1)
                assign = aux_full[f"part_assign_{L}"][i]             # (N,P)
                w = aux_full[f"curv_weight_{L}"][i]                  # (N,)
                num_parts = assign.shape[-1]

                curv_grid = token_to_grid(curv).numpy()
                assign_grid = token_to_grid(assign.argmax(-1)).numpy()

                plot_curvature(img, curv_grid,
                               os.path.join(args.out, f"curvature_layer{L}_img{n_imgs}.png"))
                plot_part_assign(img, assign_grid,
                                 os.path.join(args.out, f"partassign_layer{L}_img{n_imgs}.png"),
                                 num_parts)
                plot_curv_weight(w, os.path.join(args.out, f"curvweight_layer{L}_img{n_imgs}.png"))
                print(f"[save] img{n_imgs}: curvature/partassign/curvweight 已保存")
                n_imgs += 1
            if n_imgs >= args.num_images or n_batch >= args.max_batches * cfg.DATA.BATCH_SIZE:
                break

    if attr_attn_sum is not None:
        attr_attn_mean = attr_attn_sum / n_batch
        plot_attr_attn(attr_attn_mean.numpy(),
                       os.path.join(args.out, f"attr_attn_layer{L}.png"), attr_attn_mean.shape[0])
        print(f"[save] attr_attn_layer{L}.png (平均 {n_batch} 张图)")

    print(f"\n完成！输出目录：{os.path.abspath(args.out)}")
    print("提示：若 part_gen_1/2 的 token 数不是平方数，脚本会报错，"
          "可用 --layer 换另一层，或手动指定空间尺寸。")


if __name__ == "__main__":
    main()
