"""Launch matched multi-seed ViT controls across disjoint GPU groups."""

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path


MODE_TO_MODEL = {
    "baseline": ("vit", None),
    "part_control": ("curvpart_vit", "no_curvature"),
    "gradient_teacher": ("curvpart_vit", "gradient_teacher"),
    "entropy_teacher": ("curvpart_vit", "entropy_teacher"),
    "full": ("curvpart_vit", "full"),
    "no_hvp": ("curvpart_vit", "no_hvp"),
}


def parse_args():
    parser = argparse.ArgumentParser("Parallel ViT ablation launcher")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--category-bank", required=True)
    parser.add_argument("--output-root", default="outputs/vit_controls")
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--gpus-per-job", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=tuple(MODE_TO_MODEL),
        default=(
            "baseline",
            "part_control",
            "gradient_teacher",
            "entropy_teacher",
            "full",
        ),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone", default="vit_base_patch16_384")
    parser.add_argument("--base-port", type=int, default=29600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_command(args, mode, seed, slot):
    model, ablation = MODE_TO_MODEL[mode]
    output = Path(args.output_root) / f"{mode}_seed{seed}"
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.gpus_per_job}",
        f"--master_port={args.base_port + slot}",
        "-m",
        "vit_experiments.train",
        "--model",
        model,
        "--data-path",
        args.data_path,
        "--output",
        str(output),
        "--backbone",
        args.backbone,
        "--pretrained",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--accum-steps",
        str(args.accum_steps),
        "--workers",
        str(args.workers),
        "--seed",
        str(seed),
        "--eval-every",
        "0",
    ]
    if model == "curvpart_vit":
        command.extend(
            [
                "--category-bank",
                args.category_bank,
                "--ablation",
                ablation,
                "--insert-layers",
                "8",
                "10",
                "--num-parts",
                "8",
                "--hvp-samples",
                "4",
            ]
        )
    return output, command


def main():
    args = parse_args()
    if args.gpus_per_job < 1 or len(args.gpus) < args.gpus_per_job:
        raise ValueError("Not enough GPUs for one job")
    if len(args.gpus) % args.gpus_per_job:
        raise ValueError("GPU count must be divisible by gpus-per-job")
    groups = [
        args.gpus[index : index + args.gpus_per_job]
        for index in range(0, len(args.gpus), args.gpus_per_job)
    ]
    jobs = [(mode, seed) for mode in args.modes for seed in args.seeds]
    if args.dry_run:
        for job_index, (mode, seed) in enumerate(jobs):
            slot = job_index % len(groups)
            _, command = make_command(args, mode, seed, slot)
            prefix = f"CUDA_VISIBLE_DEVICES={','.join(map(str, groups[slot]))}"
            print(prefix, shlex.join(command))
        return

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    active = {}
    pending = list(jobs)
    while pending or active:
        for slot, group in enumerate(groups):
            if slot in active or not pending:
                continue
            mode, seed = pending.pop(0)
            output, command = make_command(args, mode, seed, slot)
            output.mkdir(parents=True, exist_ok=True)
            log_handle = open(output / "launcher.log", "a", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, group))
            print(
                f"Starting {mode} seed={seed} on GPUs {group}: "
                f"{shlex.join(command)}",
                flush=True,
            )
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            active[slot] = (process, log_handle, mode, seed)

        time.sleep(2)
        for slot, (process, log_handle, mode, seed) in list(active.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            del active[slot]
            if return_code != 0:
                for other_process, other_log, _, _ in active.values():
                    other_process.terminate()
                    other_log.close()
                raise SystemExit(
                    f"{mode} seed={seed} failed with exit code {return_code}; "
                    "inspect its launcher.log"
                )
            print(f"Finished {mode} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
