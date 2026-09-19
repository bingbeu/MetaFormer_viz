import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .MBConv import MBConvBlock
from .MHSA import MHSABlock,Mlp
from .meta_encoder import ResNormLayer
from .SemanticPartTokenGeneratorV4 import SemanticPartTokenGeneratorV4
import torch.nn.functional as F
try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'MetaFG_0': _cfg(),
    'MetaFG_1': _cfg(),
    'MetaFG_2': _cfg(),
}

def make_blocks(stage_index,depths,embed_dims,img_size,dpr,extra_token_num=1,num_heads=8,mlp_ratio=4.,stage_type='conv'):
    stage_name = f'stage_{stage_index}'
    blocks = []
    for block_idx in range(depths[stage_index]):
        stride = 2 if block_idx == 0 and stage_index != 1 else 1
        in_chans = embed_dims[stage_index] if block_idx != 0 else  embed_dims[stage_index-1]
        out_chans = embed_dims[stage_index]
        image_size = img_size if block_idx == 0 or stage_index == 1 else img_size//2
        drop_path_rate = dpr[sum(depths[1:stage_index])+block_idx]
        if stage_type == 'conv':
            blocks.append(MBConvBlock(ksize=3,input_filters=in_chans,output_filters=out_chans,
                                      image_size=image_size,expand_ratio=int(mlp_ratio),stride=stride,drop_connect_rate=drop_path_rate))
        elif stage_type == 'mhsa':
            blocks.append(MHSABlock(input_dim=in_chans,output_dim=out_chans,
                                    image_size=image_size,stride=stride,num_heads=num_heads,extra_token_num=extra_token_num,
                                    mlp_ratio=mlp_ratio,drop_path=drop_path_rate))
        else:
            raise NotImplementedError("We only support conv and mhsa")
    return blocks


"""
SemanticPartTokenGenerator — V2-fixed
=====================================
基于 V2（有界增益/中心化/概率化 align/回归分支解耦的加固版）修复"门控学不开"问题。

修复点（相对 V2 的 diff，全部标注 [FIX]）：
  1. [FIX-1] 门控初始化从"饱和区冷启动"改为"半开热启动"：
       curv_feat_alpha: -5.0 -> 0.1   (sigmoid 0.0067 -> 0.525)
       curv_logit_alpha: 0.0  -> 0.1   (tanh 0.0 -> 0.0997)
       sim_logit_alpha : 0.0  -> 0.1
       curv_sem_alpha   : 0.0  -> 0.1
     原因：实测该训练管线（lr=6.25e-6 + grad_clip=5 + AMP O1）下，
     门控标量梯度被压到 ~0（sigmoid 饱和区梯度×0.0066、grad clip 再砍 3-4 倍），
     257 epochs 后 curv_feat_alpha 停在 -4.98，曲率机制从未生效。
  2. [FIX-2] 回归分支兼容原生 torch autocast（原来只处理了 Apex disable_casts），
     防止 fp16 下 curv_reg 分支产生 inf 梯度尖峰。
  3. [FIX-3] 新增 gate_status()，训练循环每 N 个 epoch 打印一次，
     避免再出现"训练完才发现门控没开"。
  4. 保留 V2 全部其余设计（clamp 8 / gain 0.25 / 中心化 / prob 对齐 / reg 解耦）。

配套建议（在训练脚本里做，不在本文件内）：
  - 把 5 个 alpha 放进独立的 optimizer param group：
        lr *= 10~100（如 6.25e-5），weight_decay = 0
    因为实测 V1 的门控也几乎不学（0.1 -> 0.1009），不给单独 lr 门控永远学不动。
  - 若仍想"渐进式开启"，用显式 schedule：前 20 epoch 把 alpha 从 0 线性升温到 0.5，再放开，
    不要依赖优化器从饱和区自己爬出来。
  - curv_reg_weight 可从 0.1 降到 0.05（V2 的 curv_reg_loss 因学生输入无语义上下文拟合失败，
    一直贡献 ~0.5-1.0 的 loss；开 curv_sem_alpha 后应显著下降，可再观察）。
"""

class SemanticPartTokenGeneratorV3(nn.Module):
    'V3'
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_parts: int,
        attn_drop: float = 0.0,
        enable_hvp: bool = True,
        assign_scale: float = 5.0,
        curv_tau: float = 1.0,
        hvp_probe: str = "rademacher",
        hvp_samples: int = 4,
        curv_norm_eps: float = 1e-4,
        curv_reg_weight: float = 0.05,          # [V3-3] 0.1 -> 0.05
        curv_weight_max: float = 8.0,
        feat_gain_max: float = 0.25,
    ):
        super().__init__()
        if curv_tau <= 0:
            raise ValueError("curv_tau must be positive")
        if hvp_samples < 1:
            raise ValueError("hvp_samples must be at least 1")

        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_parts = num_parts
        self.enable_hvp = enable_hvp
        self.assign_scale = assign_scale
        self.curv_tau = max(curv_tau, 0.1)          # 温度下限，防 softmax 坍缩
        self.hvp_probe = hvp_probe
        self.hvp_samples = hvp_samples
        self.curv_norm_eps = curv_norm_eps
        self.curv_reg_weight = curv_reg_weight
        self.curv_weight_max = curv_weight_max
        self.feat_gain_max = feat_gain_max
        self.scale = embed_dim ** -0.5
        self.eps = 1e-6

        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.semantic_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        hidden_dim = max(embed_dim // 4, 16)
        self.curv_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self.part_queries = nn.Parameter(torch.zeros(1, num_parts, embed_dim))
        self.attn_drop = nn.Dropout(attn_drop)

        # [V3-2] 关键门控从"半开 0.1"提升到"近全强度"：
        #   曲率偏置 1.0 (tanh≈0.76)，语义相似度偏置 0.5，语义注入 0.5。
        #   实测证明曲率信号在 attention 偏置上必须全强度才有增益（V0 无门控全强度 > V1 的 0.1）。
        self.curv_feat_alpha = nn.Parameter(torch.tensor(0.1))   # 特征增益 sigmoid(0.1)≈0.525
        self.curv_sem_alpha = nn.Parameter(torch.tensor(0.5))   # 语义注入 tanh(0.5)≈0.46
        self.curv_logit_alpha = nn.Parameter(torch.tensor(1.0)) # 曲率偏置 tanh(1.0)≈0.76
        self.sim_logit_alpha = nn.Parameter(torch.tensor(0.5))  # 相似度偏置 tanh(0.5)≈0.46
        self.attr_logit_alpha = nn.Parameter(torch.tensor(1.0))
        trunc_normal_(self.part_queries, std=0.02)

    # ------------------------------------------------------------------ utils
    def gate_status(self):
        """返回门控实际生效值，供训练循环打点监控。"""
        with torch.no_grad():
            return {
                "curv_feat_gain": (self.feat_gain_max * self.curv_feat_alpha.sigmoid()).item(),
                "curv_sem_gate": self.curv_sem_alpha.tanh().item(),
                "curv_logit_gate": self.curv_logit_alpha.tanh().item(),
                "sim_logit_gate": self.sim_logit_alpha.tanh().item(),
                "attr_scale": (0.1 + F.softplus(self.attr_logit_alpha)).item(),
            }

    def _expand_token(self, token, batch_size, device, dtype):
        if token.dim() == 2:
            token = token.unsqueeze(1)
        token = token.to(device=device, dtype=dtype)
        if token.shape[0] == 1 and batch_size > 1:
            token = token.expand(batch_size, -1, -1)
        return token

    def _flatten_input(self, x):
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        elif x.dim() != 3:
            raise ValueError(f"Expected x to be 3D or 4D, got shape {tuple(x.shape)}")
        return x

    def _part_attribute_semantics(self, attr_tokens, batch_size):
        part_q = self.part_queries.expand(batch_size, -1, -1)
        sem_k = self.semantic_proj(attr_tokens)
        part_q_norm = F.normalize(part_q, dim=-1)
        sem_k_norm = F.normalize(sem_k, dim=-1)
        attr_logits = part_q_norm @ sem_k_norm.transpose(-2, -1)
        attr_scale = 0.1 + F.softplus(self.attr_logit_alpha)   # 处处有梯度
        attr_attn = torch.softmax(attr_logits * attr_scale, dim=-1)
        sem_per_part = attr_attn @ sem_k
        q = part_q + sem_per_part
        return q, sem_per_part, attr_attn

    def _token_part_similarity(self, x, sem_per_part):
        x_norm = F.normalize(x, dim=-1)
        sem_norm = F.normalize(sem_per_part, dim=-1)
        token_part_sim = torch.einsum("bnc,bpc->bnp", x_norm, sem_norm)
        part_assign = torch.softmax(token_part_sim * self.assign_scale, dim=-1)
        per_token_sim = (part_assign * token_part_sim).sum(dim=-1)
        return token_part_sim, per_token_sim, part_assign

    def _make_probe(self, hvp_x):
        if self.hvp_probe == "rademacher":
            return torch.empty_like(hvp_x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        if self.hvp_probe == "normal":
            return torch.randn_like(hvp_x)
        raise ValueError("hvp_probe must be 'rademacher' or 'normal'")

    def _compute_hvp_curvature(self, x, sem_per_part):
        # float32、仅训练的教师；推理用 curv_head 学生，保证前后一致。
        with torch.enable_grad():
            hvp_x = x.detach().float().requires_grad_(True)
            hvp_sem = sem_per_part.detach().float()
            _, per_token_sim, _ = self._token_part_similarity(hvp_x, hvp_sem)

            objective = per_token_sim.mean()
            grad = torch.autograd.grad(objective, hvp_x, create_graph=True)[0]

            diag_acc = torch.zeros_like(hvp_x)
            for sample_idx in range(self.hvp_samples):
                probe = self._make_probe(hvp_x)
                retain_graph = sample_idx < self.hvp_samples - 1
                hvp = torch.autograd.grad(
                    (grad * probe).sum(), hvp_x, retain_graph=retain_graph
                )[0]
                diag_acc = diag_acc + probe * hvp

            diag_estimate = diag_acc / float(self.hvp_samples)
            curvature = diag_estimate.norm(p=2, dim=-1, keepdim=True).detach()
        return curvature.to(dtype=x.dtype, device=x.device)

    def _normalize_curvature(self, curvature):
        # [V3-1] 除以 per-sample mean（V0 实测更优）：
        #   均值归一化后 curvature mean=1、典型范围 [0.2,5]，
        #   给 log1p 偏置和 softmax 权重提供真实对比度；
        #   max 归一化会压到 [0,1]，log1p∈[0,0.69]，信号被抹平。
        denom = curvature.mean(dim=1, keepdim=True).clamp_min(self.curv_norm_eps).detach()
        return curvature / denom

    # ---------------------------------------------------------------- forward
    def forward(self, x, extra_tokens, return_aux: bool = False):
        x = self._flatten_input(x)
        x = self.input_proj(x)
        B, N, C = x.shape

        if not isinstance(extra_tokens, (list, tuple)) or len(extra_tokens) == 0:
            raise ValueError("extra_tokens must be a non-empty list/tuple")

        tokens = [self._expand_token(t, B, x.device, x.dtype) for t in extra_tokens]
        attr_tokens = torch.cat(tokens[1:], dim=1) if len(tokens) > 1 else tokens[0]
        q, sem_per_part, attr_attn = self._part_attribute_semantics(attr_tokens, B)

        q = q.to(dtype=x.dtype)
        sem_per_part = sem_per_part.to(dtype=x.dtype)

        token_part_sim, per_token_sim, part_assign = \
            self._token_part_similarity(x, sem_per_part)

        token_sem = torch.einsum(
            "bnp,bpc->bnc", part_assign.float(), sem_per_part.float()
        ).to(dtype=x.dtype)

        sem_gate = self.curv_sem_alpha.tanh().to(dtype=x.dtype)

        # 分类/特征路径：保持端到端梯度
        forward_input = x + sem_gate * token_sem
        pred_curvature = self.curv_head(forward_input).clamp_min(self.eps)

        curv_reg_loss = pred_curvature.new_zeros(())
        hvp_curvature = None

        if (
            self.enable_hvp
            and self.training
            and torch.is_grad_enabled()
        ):
            # 回归分支不能把梯度传回 backbone 或语义分支
            reg_input = x.detach() + sem_gate * token_sem.detach()
            head_dtype = next(self.curv_head.parameters()).dtype
            reg_input = reg_input.to(dtype=head_dtype)

            # [V3-3] 回归分支强制 fp32：兼容 Apex 与原生 torch autocast
            def _run_reg_head():
                return self.curv_head(reg_input)

            if amp is not None and hasattr(amp, "disable_casts"):
                with amp.disable_casts():
                    pred_curvature_reg = _run_reg_head()
            else:
                try:
                    with torch.amp.autocast("cuda", enabled=False):
                        pred_curvature_reg = _run_reg_head()
                except (TypeError, RuntimeError):
                    pred_curvature_reg = _run_reg_head()

            pred_curvature_reg = pred_curvature_reg.clamp_min(self.eps)

            hvp_curvature = self._compute_hvp_curvature(x, sem_per_part)

            student_curvature = self._normalize_curvature(pred_curvature_reg)
            teacher_curvature = self._normalize_curvature(
                hvp_curvature
            ).detach().to(dtype=student_curvature.dtype)

            curv_reg_loss = F.smooth_l1_loss(student_curvature, teacher_curvature)

        # 后续使用预测曲率
        curvature = self._normalize_curvature(pred_curvature)
        curv_logits = curvature.squeeze(-1) / self.curv_tau
        curv_prob = torch.softmax(curv_logits, dim=1)                          # (B,N) Σ=1
        curv_weight = (N * curv_prob).clamp(max=self.curv_weight_max)          # (B,N) 有界

        entropy_denom = torch.log(curv_prob.new_tensor(float(max(N, 2))))
        curv_entropy = -(
            curv_prob * curv_prob.clamp_min(self.eps).log()
        ).sum(dim=1).mean() / entropy_denom

        # 对齐损失：直接用概率分布，天然归一化，与截断解耦
        align_loss = ((1.0 - per_token_sim) * curv_prob.detach()).sum(dim=1).mean()
        part_aux_loss = align_loss + self.curv_reg_weight * curv_reg_loss

        # 特征增强：中心化 + 有界增益，恒正、有界、无全局放大
        feat_gain = self.feat_gain_max * self.curv_feat_alpha.sigmoid()        # (0, 0.25)
        centered_weight = curv_weight - curv_weight.mean(dim=1, keepdim=True)  # 零均值
        weighted_x = x * (1.0 + feat_gain * centered_weight.unsqueeze(-1))
        k = self.key_proj(weighted_x)
        v = self.value_proj(weighted_x)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + self.sim_logit_alpha.tanh() * token_part_sim.transpose(1, 2)
        attn_logits = attn_logits + self.curv_logit_alpha.tanh() * torch.log1p(curvature).transpose(1, 2)
        attn = self.attn_drop(torch.softmax(attn_logits, dim=-1))
        part_tokens = attn @ v
        part_tokens = self.out_norm(
            self.out_proj(part_tokens)
            + self.part_queries.expand(B, -1, -1).to(part_tokens.dtype)
        )

        if return_aux:
            return part_tokens, {
                "align_loss": align_loss,
                "curv_reg_loss": curv_reg_loss,
                "part_aux_loss": part_aux_loss,
                "curvature": curvature.detach(),
                "hvp_curvature": (
                    None if hvp_curvature is None
                    else self._normalize_curvature(hvp_curvature).detach()
                ),
                "curv_weight": curv_weight.detach(),
                "curv_weight_max": curv_weight.max().detach(),
                "curv_weight_mean": curv_weight.mean().detach(),
                "curv_entropy": curv_entropy.detach(),
                "attr_attn": attr_attn.detach(),
                "part_assign": part_assign.detach(),
                "per_token_sim": per_token_sim.detach(),
                "gate_status": self.gate_status(),
            }
        return part_tokens

class MetaFG_Meta(nn.Module):
    def __init__(self,img_size=224,in_chans=3, num_classes=1000,
                conv_embed_dims = [64,96,192],attn_embed_dims=[384,768],
                conv_depths = [2,2,3],attn_depths = [5,2],num_heads=32,extra_token_num=3,mlp_ratio=4.,part_token_num=8,
                conv_norm_layer=nn.BatchNorm2d,attn_norm_layer=nn.LayerNorm,
                conv_act_layer=nn.ReLU,attn_act_layer=nn.GELU,
                qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.,
                add_meta=True,meta_dims=[4,3],mask_prob=1.0,mask_type='linear',
                only_last_cls=False,
                enable_hvp: bool = True,
                hvp_probe: str = "rademacher",
                hvp_samples: int = 4, #大规模数据集降2
                curv_tau: float = 1.0,
                curv_reg_weight: float = 0.1,
                use_checkpoint=False):
        super().__init__()
        self.only_last_cls = only_last_cls
        self.img_size = img_size
        self.num_classes = num_classes
        self.add_meta = add_meta
        self.meta_dims = meta_dims
        self.cur_epoch = -1
        self.total_epoch = -1
        self.mask_prob = mask_prob
        self.mask_type = mask_type
        self.attn_embed_dims = attn_embed_dims
        self.extra_token_num = extra_token_num
        if self.add_meta:
#             assert len(meta_dims)==extra_token_num-1
            for ind,meta_dim in enumerate(meta_dims):
                meta_head_1 = nn.Sequential(
                                        nn.Linear(meta_dim, attn_embed_dims[0]),
                                        nn.ReLU(inplace=True),
                                        nn.LayerNorm(attn_embed_dims[0]),
                                        ResNormLayer(attn_embed_dims[0]),
                                        ) if meta_dim > 0 else nn.Identity()
                meta_head_2 = nn.Sequential(
                                        nn.Linear(meta_dim, attn_embed_dims[1]),
                                        nn.ReLU(inplace=True),
                                        nn.LayerNorm(attn_embed_dims[1]),
                                        ResNormLayer(attn_embed_dims[1]),
                                        ) if meta_dim > 0 else nn.Identity()  
                setattr(self, f"meta_{ind+1}_head_1", meta_head_1)
                setattr(self, f"meta_{ind+1}_head_2", meta_head_2)
        # 第一种
        # self.part_gen_1 = SemanticPartTokenGenerator(
        #     in_dim=conv_embed_dims[2],
        #     embed_dim=attn_embed_dims[0],
        #     num_parts=part_token_num,
        #     attn_drop=attn_drop_rate,
        # )
        # self.part_gen_2 = SemanticPartTokenGenerator(
        #     in_dim=attn_embed_dims[0],
        #     embed_dim=attn_embed_dims[1],
        #     num_parts=part_token_num,
        #     attn_drop=attn_drop_rate,
        # )
        # 第二种
        self.part_gen_1 = SemanticPartTokenGeneratorV4(
            in_dim=conv_embed_dims[2],
            embed_dim=attn_embed_dims[0],
            num_parts=part_token_num,
            attn_drop=attn_drop_rate,
            enable_hvp=enable_hvp,
            hvp_probe=hvp_probe,
            hvp_samples=hvp_samples,
            curv_tau=curv_tau,
            curv_reg_weight=curv_reg_weight,
        )
        self.part_gen_2 = SemanticPartTokenGeneratorV4(
            in_dim=attn_embed_dims[0],
            embed_dim=attn_embed_dims[1],
            num_parts=part_token_num,
            attn_drop=attn_drop_rate,
            enable_hvp=enable_hvp,
            hvp_probe=hvp_probe,
            hvp_samples=hvp_samples,
            curv_tau=curv_tau,
            curv_reg_weight=curv_reg_weight,
        )
        stem_chs = (3 * (conv_embed_dims[0] // 4), conv_embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(conv_depths[1:]+attn_depths))]
        #stage_0
        self.stage_0 = nn.Sequential(*[
                nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                conv_norm_layer(stem_chs[0]),
                conv_act_layer(inplace=True),
                nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                conv_norm_layer(stem_chs[1]),
                conv_act_layer(inplace=True),
                nn.Conv2d(stem_chs[1], conv_embed_dims[0], 3, stride=1, padding=1, bias=False)])
        self.bn1 = conv_norm_layer(conv_embed_dims[0])
        self.act1 = conv_act_layer(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #stage_1
        self.stage_1 = nn.ModuleList(make_blocks(1,conv_depths+attn_depths,conv_embed_dims+attn_embed_dims,img_size//4,
                                      dpr=dpr,num_heads=num_heads,extra_token_num=extra_token_num,mlp_ratio=mlp_ratio,stage_type='conv'))
        #stage_2
        self.stage_2 = nn.ModuleList(make_blocks(2,conv_depths+attn_depths,conv_embed_dims+attn_embed_dims,img_size//4,
                                      dpr=dpr,num_heads=num_heads,extra_token_num=extra_token_num,mlp_ratio=mlp_ratio,stage_type='conv'))
        
        #stage_3
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[0]))
        self.stage_3 = nn.ModuleList(make_blocks(3,conv_depths+attn_depths,conv_embed_dims+attn_embed_dims,img_size//8,
                                      dpr=dpr,num_heads=num_heads,extra_token_num=extra_token_num,mlp_ratio=mlp_ratio,stage_type='mhsa'))
        #stage_4
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, attn_embed_dims[1]))
        self.stage_4 = nn.ModuleList(make_blocks(4,conv_depths+attn_depths,conv_embed_dims+attn_embed_dims,img_size//16,
                                      dpr=dpr,num_heads=num_heads,extra_token_num=extra_token_num,mlp_ratio=mlp_ratio,stage_type='mhsa'))
        self.norm_2 = attn_norm_layer(attn_embed_dims[1])
        
        #Aggregate
        if not self.only_last_cls:
            self.cl_1_fc = nn.Sequential(*[Mlp(in_features=attn_embed_dims[0], out_features=attn_embed_dims[1]),
                                         attn_norm_layer(attn_embed_dims[1])])
            self.aggregate = torch.nn.Conv1d(in_channels=2, out_channels=1, kernel_size=1)
            self.norm = attn_norm_layer(attn_embed_dims[1])
            self.norm_1 = attn_norm_layer(attn_embed_dims[0])
        # Classifier head
        self.curv_align_loss = None
        self.head = nn.Linear(attn_embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.cls_token_1, std=.02)
        trunc_normal_(self.cls_token_2, std=.02)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token_1','cls_token_2'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, meta=None, return_aux=False):
        B = x.shape[0]

        extra_tokens_1 = [self.cls_token_1]
        extra_tokens_2 = [self.cls_token_2]

        if self.add_meta:
            assert meta is not None, 'meta is None'

            if len(self.meta_dims) > 1:
                metas = torch.split(meta, self.meta_dims, dim=1)
            else:
                metas = (meta,)

            for ind, cur_meta in enumerate(metas):
                meta_head_1 = getattr(self, f"meta_{ind + 1}_head_1")
                meta_head_2 = getattr(self, f"meta_{ind + 1}_head_2")

                meta_1 = meta_head_1(cur_meta)
                meta_1 = meta_1.reshape(B, -1, self.attn_embed_dims[0])

                meta_2 = meta_head_2(cur_meta)
                meta_2 = meta_2.reshape(B, -1, self.attn_embed_dims[1])

                extra_tokens_1.append(meta_1)
                extra_tokens_2.append(meta_2)

        x = self.stage_0(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        for blk in self.stage_1:
            x = blk(x)

        for blk in self.stage_2:
            x = blk(x)

        out_1 = self.part_gen_1(x, extra_tokens_1, return_aux=return_aux)
        if return_aux:
            part_tokens_1, aux_1 = out_1
        else:
            part_tokens_1 = out_1

        extra_tokens_1.append(part_tokens_1)

        H0, W0 = self.img_size // 8, self.img_size // 8

        for ind, blk in enumerate(self.stage_3):
            if ind == 0:
                x = blk(x, H0, W0, extra_tokens_1)
            else:
                x = blk(x, H0, W0)

        if not self.only_last_cls:
            cls_1 = x[:, :1, :]
            cls_1 = self.norm_1(cls_1)
            cls_1 = self.cl_1_fc(cls_1)

        x = x[:, self.extra_token_num:, :]

        H1, W1 = self.img_size // 16, self.img_size // 16

        out_2 = self.part_gen_2(x, extra_tokens_2, return_aux=return_aux)
        if return_aux:
            part_tokens_2, aux_2 = out_2
        else:
            part_tokens_2 = out_2

        extra_tokens_2.append(part_tokens_2)

        x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        for ind, blk in enumerate(self.stage_4):
            if ind == 0:
                x = blk(x, H1, W1, extra_tokens_2)
            else:
                x = blk(x, H1, W1)

        cls_2 = x[:, :1, :]
        cls_2 = self.norm_2(cls_2)

        if not self.only_last_cls:
            cls = torch.cat((cls_1, cls_2), dim=1)
            cls = self.aggregate(cls).squeeze(dim=1)
            cls = self.norm(cls)
        else:
            cls = cls_2.squeeze(dim=1)

        if return_aux:
            align_loss = 0.5 * (aux_1["align_loss"] + aux_2["align_loss"])
            curv_reg_loss = 0.5 * (aux_1["curv_reg_loss"] + aux_2["curv_reg_loss"])
            # 合并后的 aux：内部已是 align + curv_reg_weight * reg（模块里 curv_reg_weight=0.05）
            part_aux_loss = 0.5 * (aux_1["part_aux_loss"] + aux_2["part_aux_loss"])
            return cls, align_loss, curv_reg_loss, part_aux_loss
        return cls


    def forward(self, x, meta=None, return_aux=False):
        if meta is not None:
            if self.mask_type == "linear":
                cur_mask_prob = (
                    self.mask_prob - self.cur_epoch / self.total_epoch
                )
            else:
                cur_mask_prob = self.mask_prob

            cur_mask_prob = max(0.0, float(cur_mask_prob))

            if cur_mask_prob != 0 and self.training:
                mask = torch.ones_like(meta)
                mask_num = int(meta.size(0) * cur_mask_prob)
                mask_index = torch.randperm(
                    meta.size(0), device=meta.device
                )[:mask_num]
                mask[mask_index] = 0
                meta = mask * meta

        if return_aux:
            feat, align_loss, curv_reg_loss, part_aux_loss = self.forward_features(
                x, meta, return_aux=True
            )
            return self.head(feat), align_loss, curv_reg_loss, part_aux_loss

        feat = self.forward_features(x, meta, return_aux=False)
        return self.head(feat)

@register_model
def MetaFG_meta_0(pretrained=False, **kwargs):
    model = MetaFG_Meta(conv_embed_dims = [64,96,192],attn_embed_dims=[384,768],
                 conv_depths = [2,2,3],attn_depths = [5,2],num_heads=8,mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['MetaFG_0']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
@register_model
def MetaFG_meta_1(pretrained=False, **kwargs):
    model = MetaFG_Meta(conv_embed_dims = [64,96,192],attn_embed_dims=[384,768],
                 conv_depths = [2,2,6],attn_depths = [14,2],num_heads=8,mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['MetaFG_1']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
@register_model
def MetaFG_meta_2(pretrained=False, **kwargs):
    model = MetaFG_Meta(conv_embed_dims = [128,128,256],attn_embed_dims=[512,1024],
                 conv_depths = [2,2,6],attn_depths = [14,2],num_heads=8,mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['MetaFG_2']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
if __name__ == "__main__":
    x = torch.randn([2, 3, 224, 224])
    meta = torch.randn([2,7])
    model = MetaFG_meta()
    import ipdb;ipdb.set_trace()
    output = model(x,meta)
    print(output.shape)