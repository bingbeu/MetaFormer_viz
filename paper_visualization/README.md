# Curv-Part Visual Analysis v2

该目录用于生成论文中的 **Visual Analysis**，与训练、评估和已有的
`fig:distillation_fidelity` 分工明确：

- `fig:distillation_fidelity`：在完整测试集上证明 Student 学习到 HVP Teacher；
- 本目录：展示部署时的 Student 在哪里产生重要性，以及 Part queries 对哪些视觉证据
  形成共识或分工。

## 为什么不再输出旧版三张图

旧版的单样本 `teacher_student`、位置编号式 `semantic_grounding` 和固定 `[0,1]`
范围的 `part_overlap` 不适合放入论文正文：

1. 单样本 HVP 带有随机探针噪声，不能代替完整测试集的蒸馏保真度统计；
2. `T01--T32` 不能说明具体词义，并且没有可靠排除 `[CLS]/[SEP]/[PAD]`；
3. 当 Part overlap 全部接近 1 时，固定 `[0,1]` 色轴会画成一块纯色，既不可读也会
   掩盖 query 冗余；
4. 对大量相同最小值做普通排序会人为产生空间梯度。v2 不再使用这种 rank 画法。

旧文件只用于排查，不应进入论文：

```text
layer*_teacher_student.*
layer*_semantic_grounding.*
layer*_part_overlap.*
```

## v2.1：同一次运行输出三套图

`visualize_evidence_consensus.py` 默认不计算 HVP，只使用部署阶段真实存在的 Student
和最终 Part attention，输出：

- `visual_analysis_main_layer2.png/.pdf`：三列正文候选版，依次为输入、Student
  importance、共享 Part evidence；
- `visual_analysis_with_query_consensus_layer2.png/.pdf`：四列完整分析版，在前三列后
  增加 `Top-10% query consensus`；
- `query_consensus_layer2.png/.pdf`：两列放大比较版，只显示输入与 query consensus，
  用于判断该信息是否值得进入正文；
- `sample_XXXX_layer2_maps.npz`：原始图与计算后的二维数组；
- `visual_analysis_metrics.json`：重叠、峰值、警告和允许/禁止的论文表述；
- 加 `--diagnostics` 后额外输出 8 个 Part queries 和 query divergence，建议放补充材料
  或用于检查，不默认放正文。

Part attention 按以下方式显示：

```text
evidence_p(i) = max(A_p(i) - 1/N, 0)
```

即只显示高于均匀注意力 `1/N` 的正增益，避免把近似均匀的 softmax 背景画得很亮。

`Top-10% query consensus` 定义为：对每个 query 取响应最高的 10% 位置，再统计每个
位置被多少比例的 queries 同时选中。它只编码**空间共识**，不编码注意力绝对强度，
因此必须与 `Shared part evidence` 联合解释，不能单独证明该区域具有很强的分类贡献。

## 1. 更新代码后先自检

```bash
cd /raid/viz/MetaFormer_viz

python3 paper_visualization/self_check_v2.py \
  --output-dir paper_visualization/outputs/self_check_v2
```

成功时应看到：

```text
[PASS] tied-value, attention-reference, metric, colour, and render checks
```

## 2. 正文候选图：Layer 2，多固定样本

Layer 2 更接近最终分类器使用的 Part tokens，正文优先使用这一层。样本编号必须在看图
前固定，不能根据效果反复挑选。下面编号只是可复现示例；正式论文应固定一组覆盖不同
类别和背景复杂度的测试样本。

```bash
cd /raid/viz/MetaFormer_viz

CUDA_VISIBLE_DEVICES=2 python3 \
  paper_visualization/visualize_evidence_consensus.py \
  --cfg configs/MetaFG_meta_bert_1_224.yaml \
  --checkpoint output/MetaFG_meta_2/cub-200-vis/best.pth \
  --dataset cub-200 \
  --data-root /raid/datasets/cub-200 \
  --sample-indices 0 1000 2000 3000 \
  --caption-index 0 \
  --img-size 384 \
  --layer 2 \
  --device cuda:0 \
  --output-dir paper_visualization/outputs/paper_layer2
```

这条命令不会重复计算 HVP Teacher，因为 Teacher--Student fidelity 已由完整测试集实验
负责。它也不会修改模型参数或 checkpoint。

运行一次后直接比较以下三个 PDF：

```text
paper_visualization/outputs/paper_layer2/visual_analysis_main_layer2.pdf
paper_visualization/outputs/paper_layer2/visual_analysis_with_query_consensus_layer2.pdf
paper_visualization/outputs/paper_layer2/query_consensus_layer2.pdf
```

正文选择原则：

- 若论文只需要说明 Student importance 如何传递为最终共享证据，优先三列版；
- 若正文主张“多个 queries 共同确认同一区域”，使用四列版，但必须写成
  `query consensus/redundancy`，不能写成 `part diversity`；
- 两列放大版主要用于作者检查或补充材料，不建议单独作为正文主图，因为 top-k
  consensus 不包含响应幅值。

## 3. Layer 1 与 8-query 诊断

```bash
CUDA_VISIBLE_DEVICES=2 python3 \
  paper_visualization/visualize_evidence_consensus.py \
  --cfg configs/MetaFG_meta_bert_1_224.yaml \
  --checkpoint output/MetaFG_meta_2/cub-200-vis/best.pth \
  --dataset cub-200 \
  --data-root /raid/datasets/cub-200 \
  --sample-indices 0 \
  --caption-index 0 \
  --img-size 384 \
  --layer 1 \
  --device cuda:0 \
  --diagnostics \
  --output-dir paper_visualization/outputs/diagnostic_layer1
```

## 如何解释 overlap

脚本计算注意力质量重叠：

```text
overlap(p,q) = sum_i min(A_p(i), A_q(i))
```

解释边界：

- overlap 较低：不同 queries 可能存在空间分工，但仍需跨样本验证；
- overlap 较高：queries 共享同一候选证据；
- `mean_pairwise_overlap >= 0.98`：脚本标记为 `high_query_redundancy`。此时只能写
  “multiple queries consistently concentrate on shared discriminative evidence”，不能写
  “the queries discover distinct/complementary anatomical parts”。

你上传的 Layer-1 样本为 `0.997496`，8 个峰值都在 `(14,13)`。这是真实的强共识，
同时也是明显的 query 冗余。它可以用来说明眼部/喙部证据被多个 queries 重复确认，
但不能证明 8 个 Part 学到了不同区域。

## 自动论文质量告警

`visual_analysis_metrics.json` 与终端日志额外给出：

- `mean_normalized_attention_entropy`：越接近 1，注意力整体越扩散；
- `student_part_consensus_spearman`：Student importance 与共享 Part evidence 的
  tie-aware Spearman 空间相关；
- `paper_warnings`：误分类、过度扩散、Student--Part 空间一致性弱、query 高冗余；
- `positive_main_text_ready`：仅是保守的筛查标志，不能代替人工核验。

误分类样本可以放 failure cases，但不能作为正文中的成功案例。样本编号应按预先声明的
规则固定（例如固定索引或类别分层抽样），不要看完热图后只挑最漂亮的样本。

## 顶刊正文还应补的一张因果可视化

现有图回答“模型看哪里”，但不能单凭热图证明这些位置导致预测。最有价值的补充是同一
图像、同一 checkpoint 的四列对照：

```text
Input | Full curvature | Uniform curvature | Shuffled curvature
```

并在每列同时报告目标类别概率变化。该图应与数据集级 Uniform/Shuffle 准确率表配套。
如果 Full、Uniform、Shuffle 的概率和空间证据几乎相同，应如实报告干预效应较弱，不能
仅凭颜色差异宣称因果作用。`fig:distillation_fidelity` 已经回答 Student 是否学习到
Teacher，不需要再增加单图 Teacher--Student 热图。

## 颜色和排版自检

- evidence：`inferno`；agreement：`viridis`；divergence：`cividis`；
- 不使用 `jet/rainbow`；
- 同一张图的 8 个 Part 使用同一归一化定义和统一色标；
- 输出双栏宽 `7.16 in`、PNG 300 DPI 和带可编辑文字的 PDF；
- 热力图只影响高响应位置，低响应处保持透明；
- 颜色仅作为强度编码，不用颜色暗示类别或统计显著性。

## 可直接放入论文的描述（需由多样本结果支持）

```latex
\paragraph{Visual analysis.}
Figure~\ref{fig:visual_analysis} visualizes the inference-time student
importance and the spatial evidence aggregated by the learned part queries.
The student responses identify candidate fine-grained cues, while the
top-response query-consensus maps reveal whether multiple queries repeatedly
select the same locations. For highly overlapping queries, we interpret the response as
evidence consensus rather than part diversity: several queries confirm the
same discriminative region, but this does not imply that they recover distinct
anatomical parts. This interpretation is consistent with the causal
Uniform/Shuffle interventions, which evaluate whether the spatial arrangement
of the predicted importance is functionally used by the classifier.
```

建议图注：

```latex
\caption{Visual analysis of inference-time evidence. From left to right:
input image, student-predicted token importance, part evidence above the
uniform-attention reference, and the fraction of part queries selecting each
location among their top 10\% responses. High query consensus is interpreted as
shared evidence consensus, not necessarily as distinct part discovery.}
```
