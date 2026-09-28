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
arguments to `args.json` under the selected output directory. By default the
test split is evaluated only at the fixed final epoch (`--eval-every 0`) and is
never used to select a checkpoint. The final checkpoint is `final.pth`. Use
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
  --backbone vit_base_patch16_384 \
  --category-bank /raid/datasets/cub-200/category_embeddings.npy \
  --checkpoint outputs/vit_curvpart_category/final.pth --eval

torchrun --standalone --nproc_per_node=2 -m vit_experiments.train \
  --model curvpart_vit --data-path "$CUB_ROOT" --output outputs/vit_curvpart_category \
  --backbone vit_base_patch16_384 --resume outputs/vit_curvpart_category/last.pth
```

When evaluating a text-conditioned checkpoint, pass the same `--semantic-root` and
`--category-bank` arguments used for training. Report at least three seeds for the
final comparison; keep every argument except `--model` and output path identical.

## Matched curvature controls

Use `--ablation no_hvp` to keep the first-order importance student while removing
its training-only HVP supervision. Use `--ablation no_curvature` for a
parameter-matched Part-Token control with uniform token importance: it disables
the HVP teacher, curvature regression, feature modulation, curvature weighting,
and the token-dependent curvature attention bias. The default `--ablation full`
is unchanged.

Two teacher controls use exactly the same architecture, semantic bank, losses,
and training schedule as the full model:

- `--ablation gradient_teacher`: the target is the token-wise channel norm of
  the gradient of the same visual--semantic compatibility objective;
- `--ablation entropy_teacher`: if
  `p_i = softmax_i(e_i / tau)` is compatibility normalized over image tokens,
  the target is the token-wise spatial-entropy contribution `-p_i log p_i`.

These definitions are implemented in `SemanticPartTokenGeneratorV6` and logged
through `curv_reg_loss`. For paper-facing comparisons, always pass the same
frozen `--category-bank` to every Curv-Part variant. Omitting it creates a
learned random bank and prints a warning because that protocol does not match
the frozen-bank paper setting.

## Recommended three-seed matrix on eight GPUs

The launcher uses four disjoint two-GPU groups, keeps the effective batch size
at `16 x 2 x 2 = 64`, and queues the remaining jobs automatically:

```bash
python -m vit_experiments.launch_ablation_matrix \
  --data-path "$CUB_ROOT" \
  --category-bank /raid/datasets/cub-200/category_embeddings.npy \
  --output-root outputs/vit_controls \
  --gpus 0 1 2 3 4 5 6 7 --gpus-per-job 2 \
  --seeds 42 43 44 --epochs 100 --batch-size 16 --accum-steps 2
```

The default matrix is `baseline`, `part_control`, `gradient_teacher`,
`entropy_teacher`, and `full`. Preview every command without launching jobs:

```bash
python -m vit_experiments.launch_ablation_matrix \
  --data-path "$CUB_ROOT" \
  --category-bank /raid/datasets/cub-200/category_embeddings.npy \
  --dry-run
```

After all runs finish, produce mean, sample standard deviation, and paired
per-seed improvements over the baseline:

```bash
python -m vit_experiments.summarize_ablation \
  --root outputs/vit_controls --seeds 42 43 44
```

## Empirical sensitivity validation

This evaluation applies the same L2-norm perturbation to every visual token and
compares six rankings on exactly the same visual--semantic compatibility
objective: HVP teacher, distilled student, first-order gradient, entropy, actual
Part attention, and a deterministic random control. It uses two empirical
targets:

1. the symmetric absolute objective change, which measures the total local
   response to a small perturbation;
2. the absolute central second difference, which isolates empirical
   second-order sensitivity.

The distinction is important: gradient is a natural control for total local
change, whereas HVP should be evaluated primarily against the second-order
response. The evaluation never updates the model or selects a checkpoint. It
defaults to a class-stratified subset of 200 CUB test images:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vit_experiments.sensitivity_eval \
  --checkpoint outputs/vit_controls/full_seed42/final.pth \
  --data-path "$CUB_ROOT" \
  --category-bank /raid/datasets/cub-200/category_embeddings.npy \
  --max-images 200 --batch-size 4 --fd-eps 0.1 --fd-samples 8 \
  --top-fraction 0.1 --curve-fractions 0.05 0.1 0.2 0.3 \
  --bootstrap-samples 5000 \
  --output outputs/vit_controls/sensitivity_seed42.json
```

Four files are written:

- `sensitivity_seed42.json`: complete protocol, summaries, paired deltas, and
  per-image/per-layer records;
- `sensitivity_seed42_summary.csv`: publication-facing means, sample SDs, and
  bootstrap 95% confidence intervals;
- `sensitivity_seed42_per_image.csv`: image-level results after averaging the
  two insertion layers;
- `sensitivity_seed42_per_image_layer.csv`: unaggregated diagnostic records.

The top-k curve reports both response capture (the fraction of all empirical
response contained in the selected tokens) and enrichment (selected-token mean
response divided by the all-token mean). For a formal multi-seed mechanism
claim, run the same command on `full_seed42`, `full_seed43`, and `full_seed44`
with distinct output names. A single predeclared seed is acceptable only when
the result is explicitly labeled as a mechanism diagnostic rather than a
training-stability result.
