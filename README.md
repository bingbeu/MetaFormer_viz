
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
part NME/PCK, GT-part coverage, predicted-part precision, and part diversity.

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_localization.py \
  --cfg output/MetaFG_meta_2/cub-200-Curv-Part/config.json \
  --ckpt output/MetaFG_meta_2/cub-200-Curv-Part/best.pth \
  --out output/localization_cub_layer2 \
  --layer 2 \
  --map-sources curvature curv_weight part_attention \
  --batch-size 8 \
  --num-workers 4 \
  --top-fraction 0.20 \
  --bootstrap-samples 2000
```

Use `--max-images 100` for a quick smoke test. Add `hvp_curvature` to
`--map-sources` only when teacher localization is needed; it computes HVPs and
is much slower, so use batch size 1 or 2. Optional true foreground masks can be
supplied with `--mask-dir`; the directory must mirror CUB image relative paths
and use PNG files. Without masks, all foreground and IoU metrics are explicitly
box-based.

Outputs:

- `localization_per_image.csv`: one row per test image;
- `localization_summary.csv`: mean, standard deviation, and bootstrap 95% CI;
- `localization_summary.json`: metric definitions and run configuration;
- `overlay_*.jpg`: green box, cyan GT parts, and yellow predicted part peaks.
