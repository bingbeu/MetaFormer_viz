"""
explicit_images.py —— 可视化专用辅助模块。

让 visualize_token.py 支持「显式指定输入图片」，而不是只能从 config 的 val 数据集读取。
本文件只服务于可视化脚本，不修改、也不依赖任何训练代码。

用法（在 visualize_token.py 里 import）：
    from explicit_images import collect_image_paths, build_explicit_image_loader

--images 支持四种写法：
    1) 目录         /path/to/images/
    2) 单张图片     /path/to/a.jpg
    3) 路径清单     /path/to/list.txt   （每行一个图片路径）
    4) 逗号分隔     a.jpg,b.jpg,c.jpg
"""

import glob
import os

import torch
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp


def build_eval_transform(cfg):
    """与训练 eval 完全一致的预处理。

    对齐 data/build.py 的 build_transform(is_train=False)：
        Resize(int(256/224 * IMG_SIZE)) -> CenterCrop(IMG_SIZE) -> ToTensor -> Normalize
    这样显式输入图片和 config 数据集的 token 位置/数值分布完全一致。
    """
    t = []
    if cfg.DATA.IMG_SIZE > 32:
        if cfg.TEST.CROP:
            size = int((256 / 224) * cfg.DATA.IMG_SIZE)
            t.append(transforms.Resize(size, interpolation=_pil_interp(cfg.DATA.INTERPOLATION)))
            t.append(transforms.CenterCrop(cfg.DATA.IMG_SIZE))
        else:
            t.append(transforms.Resize(
                (cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE),
                interpolation=_pil_interp(cfg.DATA.INTERPOLATION)))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class ExplicitImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = labels          # None 或与 paths 等长的 list[str]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        x = self.transform(img)
        y = -1 if self.labels is None else int(self.labels[idx])
        return x, y


def collect_image_paths(arg):
    """把 --images 参数解析成有序图片路径列表。"""
    paths = []
    if os.path.isdir(arg):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp",
                "*.JPG", "*.JPEG", "*.PNG", "*.BMP")
        for ext in exts:
            paths += sorted(glob.glob(os.path.join(arg, ext)))
    elif os.path.isfile(arg):
        if arg.lower().endswith((".txt", ".list")):
            with open(arg, encoding="utf-8") as f:
                paths = [ln.strip() for ln in f if ln.strip()]
        else:
            paths = [arg]
    else:
        paths = [p.strip() for p in arg.split(",") if p.strip()]
    if not paths:
        raise ValueError("未找到任何输入图片: %r" % arg)
    return paths


def build_explicit_image_loader(paths, labels, cfg):
    """构造 (loader, image_names)。batch=1、shuffle=False，按给定顺序逐张可视化。"""
    loader = torch.utils.data.DataLoader(
        ExplicitImageDataset(paths, labels, build_eval_transform(cfg)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return loader, [os.path.basename(p) for p in paths]
