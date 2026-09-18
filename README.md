
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

#### HNSD experiment

The optional Hard-Negative Semantic Part Discrimination (HNSD) branch is
training-only. It ranks the matching image caption above the hardest caption
from a different class, using both levels of generated part tokens. Captions
and labels are gathered across DDP ranks to enlarge the negative bank. Existing
configs keep HNSD disabled, so old checkpoints and inference are unchanged.

Run the isolated CUB HNSD experiment with:

```
CUDA_VISIBLE_DEVICES=2,3 python3 -m torch.distributed.launch --nproc_per_node 2 --master_port 12348 main.py --cfg configs/MetaFG_meta_bert_1_224_hnsd.yaml --batch-size 16 --tag cub-200-hnsd --lr 5e-5 --min-lr 5e-7 --warmup-lr 5e-8 --epochs 300 --warmup-epochs 20 --dataset cub-200 --pretrain /raid/MetaFormer/pretrained_model/metafg_2_inat21_384.pth --accumulation-steps 2 --opts DATA.IMG_SIZE 384
```

The default HNSD schedule is disabled through epoch 19, linearly warmed from
epoch 20 through epoch 39, and held at weight 0.005 afterward. Training logs
report the positive/negative similarity gap, active ranking ratio, valid
negative ratio, and same-class negative ratio. The last value must remain zero.

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

## Curv-Part 论文可视化 v2

论文可视化位于 [`paper_visualization/`](paper_visualization/)。v2 将 Teacher--Student
保真度留给完整测试集的 `fig:distillation_fidelity`，不再用单个随机 HVP 样本重复
论证；正文图只展示推理阶段 Student importance、共享 Part evidence 和 query agreement。

先执行自检：

```bash
cd /raid/viz/MetaFormer_viz
python3 paper_visualization/self_check_v2.py \\
  --output-dir paper_visualization/outputs/self_check_v2
```

生成 Layer-2 多样本正文候选图：

```bash
CUDA_VISIBLE_DEVICES=2 python3 \\
  paper_visualization/visualize_evidence_consensus.py \\
  --cfg configs/MetaFG_meta_bert_1_224.yaml \\
  --checkpoint output/MetaFG_meta_2/cub-200-vis/best.pth \\
  --dataset cub-200 \\
  --data-root /raid/datasets/cub-200 \\
  --sample-indices 0 1000 2000 3000 \\
  --caption-index 0 \\
  --img-size 384 \\
  --layer 2 \\
  --device cuda:0 \\
  --output-dir paper_visualization/outputs/paper_layer2
```

完整输出说明、Layer-1 诊断命令、颜色规范及允许/禁止的论文表述见
[`paper_visualization/README.md`](paper_visualization/README.md)。
