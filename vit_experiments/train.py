import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from timm.data import create_transform
from timm.loss import SoftTargetCrossEntropy
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from vit_experiments.cub_dataset import CUB200
from vit_experiments.model import build_model


def parse_args():
    parser = argparse.ArgumentParser("Matched CUB ViT experiments")
    parser.add_argument("--model", choices=("vit", "curvpart_vit"), required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backbone", default="vit_base_patch16_384")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num-classes", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16, help="Per GPU")
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=6.25e-6)
    parser.add_argument("--warmup-start-lr", type=float, default=6.25e-9)
    parser.add_argument("--min-lr", type=float, default=6.25e-8)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")

    parser.add_argument("--insert-layers", type=int, nargs="+", default=(8, 10))
    parser.add_argument("--num-parts", type=int, default=8)
    parser.add_argument("--hvp-samples", type=int, default=4)
    parser.add_argument("--curv-tau", type=float, default=1.0)
    parser.add_argument("--curv-reg-weight", type=float, default=0.1)
    parser.add_argument("--part-loss-weight", type=float, default=0.1)
    parser.add_argument("--route-loss-weight", type=float, default=0.1)
    parser.add_argument("--semantic-root", default=None)
    parser.add_argument("--semantic-key", default="embedding_words")
    parser.add_argument("--semantic-dim", type=int, default=768)
    parser.add_argument("--max-semantic-tokens", type=int, default=32)
    parser.add_argument("--category-bank", default=None)
    return parser.parse_args()


def distributed_setup():
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, local_rank, world_size


def set_seed(seed, rank):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(input_size):
    train_transform = create_transform(
        input_size=input_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
    )
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, test_transform


def build_loaders(args, distributed):
    train_transform, test_transform = build_transforms(args.input_size)
    common = dict(
        root=args.data_path,
        semantic_root=args.semantic_root if args.model == "curvpart_vit" else None,
        semantic_key=args.semantic_key,
        semantic_dim=args.semantic_dim,
        max_semantic_tokens=args.max_semantic_tokens,
    )
    train_set = CUB200(train=True, transform=train_transform, **common)
    test_set = CUB200(train=False, transform=test_transform, **common)
    train_sampler = DistributedSampler(train_set, shuffle=True) if distributed else RandomSampler(train_set)
    test_sampler = DistributedSampler(test_set, shuffle=False) if distributed else SequentialSampler(test_set)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, sampler=train_sampler, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, sampler=test_sampler, drop_last=False, **loader_args)
    return train_loader, test_loader, train_sampler


def soft_cross_entropy(logits, target):
    if target.ndim == 1:
        return F.cross_entropy(logits, target)
    return torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()


def _cutmix_box(height, width, lam, device):
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    center_y = int(torch.randint(height, (1,), device=device).item())
    center_x = int(torch.randint(width, (1,), device=device).item())
    y1, y2 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
    x1, x2 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
    return y1, y2, x1, x2


def mix_batch(images, target, semantics, args):
    """Apply one shared Mixup/CutMix permutation to images, labels, and text."""
    one_hot = F.one_hot(target, num_classes=args.num_classes).to(images.dtype)
    one_hot = one_hot * (1.0 - args.label_smoothing) + args.label_smoothing / args.num_classes
    if args.mixup <= 0 and args.cutmix <= 0:
        return images, one_hot, semantics
    permutation = torch.randperm(images.shape[0], device=images.device)
    use_cutmix = args.cutmix > 0 and (args.mixup <= 0 or torch.rand((), device=images.device) < 0.5)
    alpha = args.cutmix if use_cutmix else args.mixup
    lam = float(np.random.beta(alpha, alpha))
    if use_cutmix:
        y1, y2, x1, x2 = _cutmix_box(images.shape[-2], images.shape[-1], lam, images.device)
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(images.shape[-2] * images.shape[-1]))
    else:
        mixed_images = images * lam + images[permutation] * (1.0 - lam)
    mixed_target = one_hot * lam + one_hot[permutation] * (1.0 - lam)
    if semantics.shape[1] > 0:
        semantics = semantics * lam + semantics[permutation] * (1.0 - lam)
    return mixed_images, mixed_target, semantics


def accuracy_counts(logits, target):
    top1 = logits.argmax(dim=1)
    _, top5 = logits.topk(5, dim=1)
    correct1 = top1.eq(target).sum()
    correct5 = top5.eq(target[:, None]).any(dim=1).sum()
    return correct1, correct5, torch.tensor(target.numel(), device=target.device)


def reduce_totals(values, distributed):
    packed = torch.stack([value.float() for value in values])
    if distributed:
        dist.all_reduce(packed)
    return packed.tolist()


@torch.no_grad()
def evaluate(model, loader, device, distributed):
    model.eval()
    loss_sum = torch.zeros((), device=device)
    correct1 = torch.zeros((), device=device)
    correct5 = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    for images, target, semantics in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        semantics = semantics.to(device, non_blocking=True)
        logits = model(images, semantics)
        batch = target.numel()
        loss_sum += F.cross_entropy(logits, target, reduction="sum")
        c1, c5, n = accuracy_counts(logits, target)
        correct1 += c1
        correct5 += c5
        count += n
    loss_sum, correct1, correct5, count = reduce_totals(
        (loss_sum, correct1, correct5, count), distributed
    )
    return {
        "loss": loss_sum / count,
        "acc1": 100.0 * correct1 / count,
        "acc5": 100.0 * correct5 / count,
    }


def lr_factor(step, warmup_steps, total_steps, min_ratio, warmup_start_ratio):
    if step < warmup_steps:
        progress = float(step) / max(1, warmup_steps - 1)
        return warmup_start_ratio + (1.0 - warmup_start_ratio) * progress
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, scaler, criterion, device, args, epoch, update_step, total_updates):
    model.train()
    raw_model = model.module if hasattr(model, "module") else model
    freeze = epoch < args.freeze_backbone_epochs

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    for iteration, (images, hard_target, semantics) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        hard_target = hard_target.to(device, non_blocking=True)
        semantics = semantics.to(device, non_blocking=True)
        mixed_images, target, semantics = mix_batch(images, hard_target, semantics, args)

        amp_context = torch.cuda.amp.autocast if not args.no_amp else nullcontext
        with amp_context():
            if args.model == "curvpart_vit":
                logits, aux = model(mixed_images, semantics, return_aux=True)
                warm = args.part_loss_weight * min(1.0, float(epoch + 1) / args.warmup_epochs)
                loss = criterion(logits, target) + warm * aux["part_aux_loss"]
                loss = loss + args.route_loss_weight * soft_cross_entropy(aux["route_logits"], target)
            else:
                logits = model(mixed_images, semantics)
                loss = criterion(logits, target)
            loss = loss / args.accum_steps

        scaler.scale(loss).backward()
        should_update = (iteration + 1) % args.accum_steps == 0 or iteration + 1 == len(loader)
        if should_update:
            scaler.unscale_(optimizer)
            if freeze:
                # Keep DDP's parameter set unchanged; suppress backbone updates by
                # clearing its gradients only for the requested warm-start epochs.
                for parameter in raw_model.backbone_parameters():
                    parameter.grad = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            factor = lr_factor(
                update_step,
                args.warmup_epochs * math.ceil(len(loader) / args.accum_steps),
                total_updates,
                args.min_lr / args.lr,
                args.warmup_start_lr / args.lr,
            )
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            update_step += 1
        running_loss += loss.detach().item() * args.accum_steps
    return running_loss / len(loader), update_step


def load_weights(model, path, optimizer=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def main():
    args = parse_args()
    if not (0.0 < args.warmup_start_lr <= args.lr):
        raise ValueError("Require 0 < warmup-start-lr <= lr")
    if not (0.0 < args.min_lr <= args.lr):
        raise ValueError("Require 0 < min-lr <= lr")
    distributed, rank, local_rank, world_size = distributed_setup()
    if not torch.cuda.is_available():
        raise RuntimeError("This training entry point requires CUDA.")
    device = torch.device("cuda", local_rank)
    set_seed(args.seed, rank)
    output = Path(args.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        with open(output / "args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2)

    train_loader, test_loader, train_sampler = build_loaders(args, distributed)
    model = build_model(args).to(device)
    if args.checkpoint:
        load_weights(model, args.checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.no_amp)
    start_epoch, update_step, best_acc = 0, 0, 0.0
    if args.resume:
        state = load_weights(model, args.resume, optimizer, scaler)
        start_epoch = state["epoch"] + 1
        update_step = state.get("update_step", 0)
        best_acc = state.get("best_acc", 0.0)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    if args.eval:
        metrics = evaluate(model, test_loader, device, distributed)
        if rank == 0:
            print(json.dumps(metrics, indent=2))
        return

    criterion = SoftTargetCrossEntropy()
    updates_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_updates = args.epochs * updates_per_epoch
    start = time.time()
    for epoch in range(start_epoch, args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        train_loss, update_step = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            args, epoch, update_step, total_updates,
        )
        metrics = evaluate(model, test_loader, device, distributed)
        if rank == 0:
            raw_model = model.module if hasattr(model, "module") else model
            record = {"epoch": epoch, "train_loss": train_loss, **metrics}
            print(json.dumps(record))
            with open(output / "log.jsonl", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            state = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "update_step": update_step,
                "best_acc": max(best_acc, metrics["acc1"]),
                "args": vars(args),
            }
            torch.save(state, output / "last.pth")
            if metrics["acc1"] > best_acc:
                best_acc = metrics["acc1"]
                state["best_acc"] = best_acc
                torch.save(state, output / "best.pth")
    if rank == 0:
        print(f"Training time: {(time.time() - start) / 3600:.2f} h; best Acc@1: {best_acc:.2f}")


if __name__ == "__main__":
    main()
