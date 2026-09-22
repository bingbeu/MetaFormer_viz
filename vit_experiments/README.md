# ViT baseline and Curv-Part on CUB-200-2011

This directory provides a matched comparison: the same ViT, official CUB split,
augmentations, optimizer, schedule, resolution, seed, and evaluation code are used
for both runs. The only experimental switch is `--model`.

## Environment

Run commands from the repository root. The project is compatible with the
repository's PyTorch/timm environment; it additionally uses NumPy and Pillow.

```bash
export CUB_ROOT=/raid/datasets/cub-200/CUB_200_2011
python -m vit_experiments.train --help
```

`--data-path` accepts either the directory above or its parent
`/raid/datasets/cub-200`. The loader always uses the official train/test split.

`--batch-size` is the per-GPU batch size. The effective batch size is
`batch-size x number of GPUs x accum-steps`; for example, the defaults give
`16 x 2 x 2 = 64` on two GPUs. Training displays a rank-zero progress bar and
writes step metrics to `train_steps.jsonl`, epoch metrics to `log.jsonl`, and
arguments to `args.json` under the selected output directory. Use
`--log-interval N` to control step-log frequency or `--no-progress` for plain
JSON terminal output.

## Baseline

Single GPU smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vit_experiments.train \
  --model vit --data-path "$CUB_ROOT" --output outputs/vit_baseline_smoke \
  --backbone vit_base_patch16_384 --pretrained --epochs 1 --workers 4
```

Matched two-GPU baseline:

```bash
torchrun --standalone --nproc_per_node=2 -m vit_experiments.train \
  --model vit --data-path "$CUB_ROOT" --output outputs/vit_baseline \
  --backbone vit_base_patch16_384 --pretrained --seed 42
```

## Curv-Part

Category-conditioned run using only the official CUB directory:

```bash
torchrun --standalone --nproc_per_node=2 -m vit_experiments.train \
  --model curvpart_vit --data-path "$CUB_ROOT" --output outputs/vit_curvpart_category \
  --backbone vit_base_patch16_384 --pretrained --seed 42 \
  --insert-layers 8 10 --num-parts 8 --hvp-samples 4
```

Without `--category-bank`, category-context prototypes are learned. Category
selection always comes from the model's route prediction; the ground-truth label
is used only by the auxiliary route loss.

To reproduce description-conditioned training with the repository's existing
precomputed CUB features and class-name embedding bank:

```bash
torchrun --standalone --nproc_per_node=2 -m vit_experiments.train \
  --model curvpart_vit --data-path "$CUB_ROOT" --output outputs/vit_curvpart_text \
  --backbone vit_base_patch16_384 --pretrained --seed 42 \
  --semantic-root /raid/datasets/cub-200/bert_embedding_cub \
  --category-bank /raid/datasets/cub-200/category_embeddings.npy \
  --insert-layers 8 10 --num-parts 8 --hvp-samples 4
```

The semantic feature tree must mirror `CUB_200_2011/images` and contain `.pickle`
files with `embedding_words` arrays of shape `[tokens, 768]`. At inference, the
training teacher is disabled automatically; only the learned first-order student
and the generated Part Tokens remain.

## Evaluation and resume

```bash
CUDA_VISIBLE_DEVICES=0 python -m vit_experiments.train \
  --model curvpart_vit --data-path "$CUB_ROOT" --output outputs/eval \
  --backbone vit_base_patch16_384 --checkpoint outputs/vit_curvpart_category/best.pth \
  --eval

torchrun --standalone --nproc_per_node=2 -m vit_experiments.train \
  --model curvpart_vit --data-path "$CUB_ROOT" --output outputs/vit_curvpart_category \
  --backbone vit_base_patch16_384 --resume outputs/vit_curvpart_category/last.pth
```

When evaluating a text-conditioned checkpoint, pass the same `--semantic-root` and
`--category-bank` arguments used for training. Report at least three seeds for the
final comparison; keep every argument except `--model` and output path identical.

## Curvature controls

Use `--ablation no_hvp` to keep the first-order importance student while removing
its training-only HVP supervision. Use `--ablation no_curvature` for a
parameter-matched Part-Token control with uniform token importance: it disables
the HVP teacher, curvature regression, feature modulation, curvature weighting,
and the token-dependent curvature attention bias. The default `--ablation full`
is unchanged.
