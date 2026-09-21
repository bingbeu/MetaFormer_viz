# Paper experiment protocol

This protocol uses the existing best checkpoint. The localization suites do
not train a new network and do not alter its classification weights.

## 1. Pull the code on the server

```bash
cd /raid/viz/MetaFormer_viz_fore
git fetch origin codex/paper-experiments
git switch -c paper-experiments --track origin/codex/paper-experiments
```

If the repository remote is named `metaformer`, replace `origin` with
`metaformer`. To update an existing local branch, use `git pull --ff-only`.

## 2. Verify the checkpoint before long runs

```bash
cd /raid/viz/MetaFormer_viz_fore
CFG=output/MetaFG_meta_2/cub-200vis/config.json
CKPT=output/MetaFG_meta_2/cub-200vis/best.pth

test -f "$CFG" && test -f "$CKPT"
CUDA_VISIBLE_DEVICES=0 python evaluate_localization.py \
  --cfg "$CFG" --ckpt "$CKPT" \
  --out output/paper_smoke --layer 2 \
  --map-sources curvature curv_weight part_attention \
  --max-images 30 --sample-mode stratified \
  --content-attention-mode raw --save-overlays 0 \
  --save-attention-maps 0 --bootstrap-samples 200
```

## 3. Recommended no-training experiments

Run the three attention definitions on all 5,794 CUB test images:

```bash
CUDA_VISIBLE_DEVICES=0 python run_paper_experiments.py \
  --cfg "$CFG" --ckpt "$CKPT" \
  --out-root output/paper_core_full \
  --suite core --max-images 0 --batch-size 8 \
  --bootstrap-samples 2000 --save-attention-maps 20
```

Run layer, top-fraction, causal deletion, flip-stability, and decomposition
checks. Start with 200 stratified images; use `--max-images 0` for the final
table. Completed directories are skipped, so the command is restart-safe.

```bash
CUDA_VISIBLE_DEVICES=0 python run_paper_experiments.py \
  --cfg "$CFG" --ckpt "$CKPT" \
  --out-root output/paper_robustness \
  --suite layers thresholds causal diagnostics \
  --max-images 200 --batch-size 8 --bootstrap-samples 2000
```

HVP teacher curvature is deliberately excluded because it is much slower.
Add `--include-hvp` only for a small diagnostic run, then repeat the selected
comparison on the full set if it is useful.

## 4. Paired confidence intervals

Never compare only two independently bootstrapped means. Pair the same images:

```bash
python compare_localization_runs.py \
  --baseline output/paper_core_full/raw_layer2 \
  --candidate output/paper_core_full/cosine_rms_layer2 \
  --out output/paper_core_full/raw_vs_cosine_rms \
  --bootstrap-samples 10000 --seed 0
```

The CSV is ready for a paper table. Report the mean paired difference and its
95% bootstrap interval. Classification Top-1/Top-5 should be unchanged for a
checkpoint-only reinterpretation; otherwise stop and inspect the forward path.

## 5. Evaluation versus training

For the paper's current checkpoint-only evidence, do **not** train. Use the
commands above. `cosine_mean_norm` and `cosine_rms` replace only the content
attention computation at inference, so label them as post-hoc variants unless
you fine-tune them.

To evaluate the original checkpoint with the normal validation path:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node 1 --master_port 12348 main.py \
  --eval --cfg "$CFG" --dataset cub-200 --resume "$CKPT" \
  --batch-size 8 --opts MODEL.CONTENT_ATTENTION_MODE raw
```

To fine-tune `cosine_rms` for 10 new epochs from the model weights (without
restoring the old optimizer/epoch), load the checkpoint as `--pretrain`, not
`--resume`. Use a new output tag so auto-resume cannot pick up the old run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node 1 --master_port 12349 main.py \
  --cfg "$CFG" --dataset cub-200 --pretrain "$CKPT" \
  --output output --tag cub-200-cosine-rms-ft10 \
  --batch-size 4 --accumulation-steps 2 \
  --epochs 10 --warmup-epochs 1 \
  --lr 5e-6 --min-lr 5e-8 --warmup-lr 5e-7 \
  --opts MODEL.CONTENT_ATTENTION_MODE cosine_rms \
         MODEL.DORP_HEAD False MODEL.DORP_META False \
         TRAIN.AUTO_RESUME False
```

First run one epoch and check that checkpoint loading reports no important
missing/unexpected keys. The 10-epoch command is a fine-tuning experiment; it
must not replace the 93.06% result unless the same evaluation protocol confirms
the new checkpoint. Full training from an external pretrained model should use
the original README recipe and a 300-epoch schedule.

## 6. What to report

- Full-set metrics, not the 200-image development subset.
- Raw versus candidate paired deltas and 95% confidence intervals.
- Fixed image sampling and fixed token IDs; qualitative examples are secondary.
- Pointing game, foreground energy/gain, pixel IoU, predicted-box IoU, and
  localization accuracy at IoU 0.5.
- Causal deletion relative to matched random foreground and background masks.
- The exact checkpoint SHA/hash, config, branch commit, seed, and command.

Do not describe box-based foreground metrics as segmentation-mask metrics. Do
not use the post-hoc cosine maps as evidence that the originally trained raw
attention learned semantic parts; they support a narrower claim about
norm-decoupled spatial evidence.
