# Curv-Part 论文可视化

本目录与原有训练、评估和可视化代码完全分离。脚本只读取训练好的检查点和
`return_aux=True` 返回的观测量，不修改模型参数，也不改变训练或分类前向。

## 输出内容

- `layer2_overview.*`：输入、Student 曲率、HVP Teacher、Part 共识和覆盖范围。
- `layer2_part_attention.*`：8 个真实 Part 空间注意力，以及 Part agreement/coverage。
- `layer2_part_overlap.*`：Part 两两注意力质量重叠系数。
- `layer2_teacher_student.*`：Teacher/Student 百分位空间图和绝对秩差异。
- `layer2_semantic_grounding.*`：Part 到 32 个描述 token 的接地权重。
- `layer2_maps.npz`：所有原始二维图，便于后续重新排版。
- `metrics.json`：预测、置信度、Teacher--Student 相关性和 Part 重叠统计。

Part 不被强制解释为互斥部件。较低的重叠表示分工，较高的重叠表示多个 Part
对同一候选区域形成共识；两种现象都可能存在。重叠本身不能单独证明判别性，
需要结合正确分类、置信度变化、曲率干预和数据集级定位指标解释。

## 1. 先执行 CPU 绘图自检

```bash
cd /raid/viz/MetaFormer_viz
python3 paper_visualization/self_check.py \
  --output-dir paper_visualization/outputs/self_check
```

成功时应看到：

```text
[PASS] plotting, metrics, output files, and colour policy
```

## 2. 生成 CUB 单样本论文图

```bash
cd /raid/viz/MetaFormer_viz

CUDA_VISIBLE_DEVICES=2 python3 \
  paper_visualization/visualize_part_evidence.py \
  --cfg configs/MetaFG_meta_bert_1_224.yaml \
  --checkpoint output/MetaFG_meta_2/cub-200vis/best.pth \
  --dataset cub-200 \
  --data-root /raid/datasets/cub-200 \
  --sample-index 0 \
  --caption-index 0 \
  --img-size 384 \
  --layer 2 \
  --device cuda:0 \
  --force-hvp \
  --hvp-layer 2 \
  --output-dir paper_visualization/outputs/cub_sample_0000
```

更换样本只需修改 `--sample-index`。正式论文中应提前固定样本编号，不能根据
热力图效果反复挑选。验证/测试 caption 建议固定 `--caption-index 0`，与正式评估
协议保持一致。

如果只需要推理阶段的 Student 和 Part 注意力，不计算 HVP Teacher，删除：

```text
--force-hvp --hvp-layer 2
```

## 3. 第一层高分辨率 Part 图

```bash
CUDA_VISIBLE_DEVICES=2 python3 \
  paper_visualization/visualize_part_evidence.py \
  --cfg configs/MetaFG_meta_bert_1_224.yaml \
  --checkpoint output/MetaFG_meta_2/cub-200vis/best.pth \
  --dataset cub-200 \
  --data-root /raid/datasets/cub-200 \
  --sample-index 0 \
  --caption-index 0 \
  --img-size 384 \
  --layer 1 \
  --device cuda:0 \
  --force-hvp \
  --hvp-layer 1 \
  --output-dir paper_visualization/outputs/cub_sample_0000_layer1
```

## 颜色与排版规范

- Attention/curvature：`magma`。
- Part agreement 与重叠矩阵：`viridis`。
- Teacher--Student 绝对秩差异：`cividis`。
- 禁止使用 `jet`/`rainbow`。
- 同一样本的 8 个 Part 共用同一颜色范围，避免独立归一化夸大差异。
- PNG 为 300 DPI，同时保存带可编辑文字的 PDF 矢量版本。
- 图宽按双栏约 `7.16 in` 输出，白底、黑字，最小字号不低于 7 pt。

## 指标解释

`mean_pairwise_overlap` 使用注意力质量重叠系数：

```text
overlap(p, q) = sum_i min(A[p, i], A[q, i])
```

取值范围 `[0, 1]`：

- 接近 0：两个 Part 的空间证据互补；
- 接近 1：两个 Part 对相同区域形成高共识；
- 中间值：同时存在共享证据和局部分工。

不能把“高重叠”直接等价为“强判别性”。更严谨的论证是：多个 Part 对某区域
形成共识，并且该区域位于目标前景、在 Full 模型中稳定出现，且 Uniform/Shuffle
干预后共识或分类置信度下降。

