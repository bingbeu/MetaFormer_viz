
## Usage
#### python module
* install `Pytorch and torchvision`
- torch==1.13.1
- torchvision==0.14.1
- transformers==4.30.2
- timm==0.4.5
```
* install `Apex`
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
* install other requirements
```
pip install opencv-python==4.5.1.48 yacs==0.1.8
```
#### data preparation
```
datasets
  |————inraturelist2021
  |       └——————train
  |       └——————val
  |       └——————train.json
  |       └——————val.json
  |————inraturelist2018
  |       └——————train_val_images
  |       └——————train2018.json
  |       └——————val2018.json
  |       └——————train2018_locations.json
  |       └——————val2018_locations.json
  |       └——————categories.json.json
  |————inraturelist2017
  |       └——————train_val_images
  |       └——————train2017.json
  |       └——————val2017.json
  |       └——————train2017_locations.json
  |       └——————val2017_locations.json
  |————cub-200
  |       └——————...
  |————nabirds
  |       └——————...
  |————stanfordcars
  |       └——————car_ims
  |       └——————cars_annos.mat
  |————aircraft
  |       └——————...
```
#### Training
You can dowmload pre-trained model from model zoo, and put them under \<root\>/pretrained.
To train MetaFG on datasets, run:
```
python3 -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py --cfg <config-file> --dataset <dataset-name> --pretrain <pretainedmodel-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```
\<dataset-name\>:inaturelist2021,inaturelist2018,inaturelist2017,cub-200,nabirds,stanfordcars,aircraft
For CUB-200-2011, run:
```
CUDA_VISIBLE_DEVICES=4,5 python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 12348  main.py --cfg /raid/MetaFormer/configs/MetaFG_meta_bert_1_224.yaml --batch-size 4 --tag cub-200_v1 --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 --epochs 300 --warmup-epochs 20 --dataset cub-200 --pretrain /raid/MetaFormer/pretrained_model/metafg_2_inat21_384.pth --accumulation-steps 2 --opts DATA.IMG_SIZE 384  
```
note that final learning rate is total_bs/512.
#### Eval
To evaluate model on dataset,run:
```
python3 -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py --eval --cfg <config-file> --dataset <dataset-name> --resume <checkpoint> [--batch-size <batch-size-per-gpu>]
```
## Main Result
#### ImageNet-1k 
| Name       | Resolution   | #Param   |  #FLOPS   | Throughput   | Top-1 acc |
| :--------: | :----------: | :--------: | :----------: | :------------: | :------------: |
| MetaFormer-0   |     224x224      |  28M  |  4.6G  |  840.1  | 82.9 |
| MetaFormer-1   |     224x224      |  45M  |  8.5G  |  444.8  | 83.9 |
| MetaFormer-2   |     224x224      |  81M  |  16.9G  |  438.9  | 84.1 |
| MetaFormer-0   |     384x384      |  28M  |  13.4G  |  349.4  | 84.2 |
| MetaFormer-1   |     384x384      |  45M  |  24.7G  |  165.3  | 84.4 |
| MetaFormer-2   |     384x384      |  81M  |  49.7G  |  132.7  | 84.6 |
#### Fine-grained Datasets
Result on fine-grained datasets with different pre-trained model.
| Name       | Pretrain   | CUB | NABirds |  iNat2017   | iNat2018  | Cars | Aircraft |
| :--------: | :----------: | :--------: | :----------: | :------------: | :------------: | :--------: |:--------: |
| MetaFormer-0|ImageNet-1k|89.6|89.1|75.7|79.5|95.0|91.2|
| MetaFormer-0|ImageNet-21k|89.7|89.5|75.8|79.9|94.6|91.2|
| MetaFormer-0|iNaturalist 2021|91.8|91.5|78.3|82.9|95.1|87.4|
| MetaFormer-1|ImageNet-1k|89.7|89.4|78.2|81.9|94.9|90.8|
| MetaFormer-1|ImageNet-21k|91.3|91.6|79.4|83.2|95.0|92.6|
| MetaFormer-1|iNaturalist 2021|92.3|92.7|82.0|87.5|95.0|92.5|
| MetaFormer-2|ImageNet-1k|89.7|89.7|79.0|82.6|95.0|92.4|
| MetaFormer-2|ImageNet-21k|91.8|92.2|80.4|84.3|95.1|92.9|
| MetaFormer-2|iNaturalist 2021|92.9|93.0|82.8|87.7|95.4|92.8|


Results in iNaturalist 2019, iNaturalist 2018, and iNaturalist 2021 with meta-information.
| Name       | Pretrain   | Meta added| iNat2017   |  iNat2018   | iNat2021   |
| :--------: | :----------: | :--------: | :---------- | :------------ |:------------ |
|MetaFormer-0|ImageNet-1k|N|75.7|79.5|88.4|
|MetaFormer-0|ImageNet-1k|Y|79.8(+4.1)|85.4(+5.9)|92.6(+4.2)|
|MetaFormer-1|ImageNet-1k|N|78.2|81.9|90.2|
|MetaFormer-1|ImageNet-1k|Y|81.3(+3.1)|86.5(+4.6)|93.4(+3.2)|
|MetaFormer-2|ImageNet-1k|N|79.0|82.6|89.8|
|MetaFormer-2|ImageNet-1k|Y|82.0(+3.0)|86.8(+4.2)|93.2(+3.4)|
|MetaFormer-2|ImageNet-21k|N|80.4|84.3|90.3|
|MetaFormer-2|ImageNet-21k|Y|83.4(+3.0)|88.7(+4.4)|93.6(+3.3)|
## Citation

```
@article{MetaFormer,
  title={MetaFormer: A Unified Meta Framework for Fine-Grained Recognition},
  author={Diao, Qishuai and Jiang, Yi and Wen, Bin and Sun, Jia and Yuan, Zehuan},
  journal={arXiv preprint arXiv:2203.02751},
  year={2022},
}
```

## Acknowledgement
Many thanks for [swin-transformer](https://github.com/microsoft/Swin-Transformer).A part of the code is borrowed from it.

#### Quantitative localization evaluation (CUB-200-2011)

`evaluate_localization.py` evaluates the deployable curvature student and the
actual pre-dropout attention used to form part tokens against the official CUB
bounding boxes and visible part keypoints. It reports pointing-game accuracy,
top-fraction pixel IoU, predicted-box IoU, IoU@0.5 localization accuracy,
foreground energy/concentration, top-k foreground precision, Hungarian-matched
part NME/PCK, GT-part coverage, and Softmax evidence-token consensus. Shared
token peaks are treated as agreement, not automatically as part collapse.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_localization.py \
  --cfg output/MetaFG_meta_2/cub-200-Curv-Part/config.json \
  --ckpt output/MetaFG_meta_2/cub-200-Curv-Part/best.pth \
  --out output/localization_cub_layer2 \
  --layer 2 \
  --map-sources curvature curv_weight part_attention \
  --batch-size 8 \
  --num-workers 4 \
  --sample-mode stratified \
  --top-fraction 0.20 \
  --save-attention-maps 20 \
  --visualize-top-k 4 \
  --visualize-token-selection fixed \
  --attention-aggregation mean \
  --attention-interpolation nearest \
  --bootstrap-samples 2000
```

To test whether a shared token peak is causally discriminative rather than a
background/position shortcut, enable fixed-area deletion and flip stability:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_localization.py \
  --cfg output/MetaFG_meta_2/cub-200-Curv-Part/config.json \
  --ckpt output/MetaFG_meta_2/cub-200-Curv-Part/best.pth \
  --out output/localization_causal_layer2 \
  --layer 2 \
  --map-sources curvature part_attention \
  --batch-size 1 \
  --causal-deletion \
  --deletion-size 0.15 \
  --deletion-random-samples 5 \
  --deletion-batch-size 4 \
  --flip-stability
```

To diagnose which additive term creates a boundary-biased Part-attention map:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_localization.py \
  --cfg output/MetaFG_meta_2/cub-200vis/config.json \
  --ckpt output/MetaFG_meta_2/cub-200vis/best.pth \
  --out output/attention_decomposition_layer2 \
  --layer 2 \
  --map-sources part_attention \
  --max-images 200 \
  --sample-mode stratified \
  --attention-decomposition \
  --save-decomposition-maps 20
```

`decomposition_*.png` shows Input, Content, Semantic, Curvature, and Final maps
from the same Part-attention logits. The CSV reports centered RMS contribution,
correlation with the final logits, peak agreement, localization metrics for
each component, and the Softmax reconstruction error. This is a diagnostic of
the existing Softmax attention, not a different attribution method.

If Content dominates, diagnose whether raw dot-product key norms create the
boundary peak. This does not alter predictions:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_localization.py \
  --cfg output/MetaFG_meta_2/cub-200vis/config.json \
  --ckpt output/MetaFG_meta_2/cub-200vis/best.pth \
  --out output/content_norm_diagnostic_layer2 \
  --layer 2 --map-sources part_attention \
  --max-images 200 --sample-mode stratified \
  --content-norm-diagnostic --save-content-norm-maps 20
```

This diagnostic requires the updated `SemanticPartTokenGeneratorV6.py` and
`MetaFG_meta.py`, which expose detached raw content logits, cosine content
logits, and key norms only when `return_aux=True`.

When the diagnostic confirms spatial key-norm bias, evaluate the real
norm-decoupled forward path with `--content-attention-mode cosine_mean_norm`.
It still uses the original spatial Softmax and does not impose token diversity.
The mean query/key norms retain a conservative per-image, per-Part-token logit
scale, while spatial ranking comes from cosine similarity:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_localization.py \
  --cfg output/MetaFG_meta_2/cub-200vis/config.json \
  --ckpt output/MetaFG_meta_2/cub-200vis/best.pth \
  --out output/cosine_mean_norm_layer2 \
  --layer 2 --map-sources part_attention \
  --max-images 200 --sample-mode stratified \
  --content-attention-mode cosine_mean_norm \
  --save-attention-maps 20
```

The default is `--content-attention-mode raw`, which is exactly the historical
checkpoint behavior. Compare classification and localization metrics between
the two output directories before deciding whether to fine-tune the model.

For normal training or `main.py --eval`, select the same forward path through
the regular configuration system:

```bash
python main.py --cfg <config.yaml> --eval --resume <checkpoint.pth> \
  --opts MODEL.CONTENT_ATTENTION_MODE cosine_mean_norm
```

`MODEL.CONTENT_ATTENTION_MODE` defaults to `raw` for backward compatibility.

For sharp, publication-facing maps, use `cosine_rms`. It removes spatial
key-norm bias like `cosine_mean_norm`, but matches every Part token's centered
raw-logit RMS so Softmax sharpness is not lost. The summary additionally
reports normalized entropy, effective-support fraction, and peak-over-uniform
ratio; these must be inspected together with localization metrics.

`evidence_consensus_ratio` measures how many evidence tokens share the modal
peak and is descriptive rather than an optimization target. A positive
`causal_consensus_minus_random_foreground_target_probability_drop` means that
masking the consensus patch hurts the target-class confidence more than masking
matched random foreground patches. Lower `evidence_flip_stability_nme` means
the consensus location is more stable after horizontal-flip inversion.

Use `--max-images 100` for a quick smoke test. With the default
`--sample-mode stratified`, those images are drawn across classes instead of
from CUB's class-sorted prefix. Add `hvp_curvature` to
`--map-sources` only when teacher localization is needed; it computes HVPs and
is much slower, so use batch size 1 or 2. Optional true foreground masks can be
supplied with `--mask-dir`; the directory must mirror CUB image relative paths
and use PNG files. Without masks, all foreground and IoU metrics are explicitly
box-based.

Outputs:

- `localization_per_image.csv`: one row per test image;
- `localization_summary.csv`: mean, standard deviation, and bootstrap 95% CI;
- `localization_summary.json`: metric definitions and run configuration;
- `attention_*.png`: input image, one aggregate Softmax evidence map, and the
  selected evidence-token maps. `--visualize-top-k` may be smaller than
  `--num-parts`; the default is the fixed indices 0,1,2,3, which avoids
  per-image cherry-picking. Use `--visualize-token-ids 0 2 5` to choose an
  explicit fixed subset. `top_peak` selection remains available only as a
  labelled diagnostic. All displayed attention panels share one zero-based
  color scale rather than independent min-max stretching. Quantitative results include
  all tokens (`part_all_tokens_*`) plus both mean/max aggregate maps
  (`part_attention_mean_*` and `part_attention_max_*`). These are evidence
  tokens, not attention heads.
