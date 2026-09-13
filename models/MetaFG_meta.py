import math
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.checkpoint as checkpoint
from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .MBConv import MBConvBlock
from .MHSA import MHSABlock,Mlp
from .meta_encoder import ResNormLayer
from .SemanticPartTokenGeneratorV6 import SemanticPartTokenGeneratorV6
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



class MetaFG_Meta(nn.Module):
    def __init__(self,img_size=224,in_chans=3, num_classes=1000,
                conv_embed_dims = [64,96,192],attn_embed_dims=[384,768],
                conv_depths = [2,2,3],attn_depths = [5,2],num_heads=32,extra_token_num=3,mlp_ratio=4.,part_token_num=8,
                conv_norm_layer=nn.BatchNorm2d,attn_norm_layer=nn.LayerNorm,
                conv_act_layer=nn.ReLU,attn_act_layer=nn.GELU,
                qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.,
                add_meta=True,meta_dims=[4,3],mask_prob=1.0,mask_type='linear',
                only_last_cls=False,assess = False,
                enable_hvp: bool = True,
                hvp_probe: str = "rademacher",
                hvp_samples: int = 4, #大规模数据集降2
                curv_tau: float = 1.0,
                curv_reg_weight: float = 0.1,
                category_emb_path: str = None,   # 类别名文本库 .npy [num_classes, 768]
                temperature: float = 1.0,         # 类别路由 softmax 温度
                lambda_route: float = 0.1,        # loss_route 权重
                enable_hnsd: bool = False,
                hnsd_margin: float = 0.10,
                use_checkpoint=False):
        super().__init__()
        self.only_last_cls = only_last_cls
        self.assess = assess
        self.img_size = img_size
        self.num_classes = num_classes
        self.add_meta = add_meta
        self.meta_dims = meta_dims
        self.cur_epoch = -1
        self.total_epoch = -1
        self.mask_prob = mask_prob
        self.mask_type = mask_type
        self.temperature = temperature
        self.lambda_route = lambda_route
        self.enable_hnsd = enable_hnsd
        self.hnsd_margin = float(hnsd_margin)
        if self.hnsd_margin < 0:
            raise ValueError("hnsd_margin must be non-negative")
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
        # 类别名文本库（冻结）：[num_classes, 768]，用于 V8-Lite 类别语义路由
        if category_emb_path is not None and os.path.exists(category_emb_path):
            bank = torch.from_numpy(np.load(category_emb_path)).float()   # [C, 768]
            self.register_buffer('category_bank', bank, persistent=False)
        else:
            self.register_buffer('category_bank', torch.zeros(0))
        # 类别语义投影头（768 → stage3/4 维度）
        self.cat_head_1 = nn.Linear(768, attn_embed_dims[0])
        self.cat_head_2 = nn.Linear(768, attn_embed_dims[1])
        # V8-Lite 类别路由头：全局图像特征 → num_classes 类 logits
        self.route_norm = nn.LayerNorm(conv_embed_dims[2])
        self.route_head = nn.Linear(conv_embed_dims[2], num_classes)
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
        self.part_gen_1 = SemanticPartTokenGeneratorV6(
            in_dim=conv_embed_dims[2],
            embed_dim=attn_embed_dims[0],
            num_parts=part_token_num,
            attn_drop=attn_drop_rate,
            enable_hvp=enable_hvp,
            hvp_probe=hvp_probe,
            hvp_samples=hvp_samples,
            curv_tau=curv_tau,
            curv_reg_weight=curv_reg_weight
        )
        self.part_gen_2 = SemanticPartTokenGeneratorV6(
            in_dim=attn_embed_dims[0],
            embed_dim=attn_embed_dims[1],
            num_parts=part_token_num,
            attn_drop=attn_drop_rate,
            enable_hvp=enable_hvp,
            hvp_probe=hvp_probe,
            hvp_samples=hvp_samples,
            curv_tau=curv_tau,
            curv_reg_weight=curv_reg_weight
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

    @staticmethod
    def _gather_detached(tensor):
        """Gather a detached tensor from all DDP ranks for a larger negative bank."""
        tensor = tensor.detach().contiguous()
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def _hnsd_stage_loss(self, part_tokens, text_tokens, targets):
        """Compute one-stage hard-negative local semantic ranking loss.

        Text tokens and the distributed negative bank are detached. Gradients
        therefore optimize the visual part representation without turning the
        caption embeddings into a shortcut. Negatives sharing the image label
        are explicitly excluded, including the image's own caption.
        """
        part_tokens_fp32 = F.normalize(part_tokens.float(), dim=-1)
        positive_text_fp32 = F.normalize(text_tokens.detach().float(), dim=-1)
        local_targets = targets.detach().long().reshape(-1)

        positive_token_scores = torch.einsum(
            "bpc,btc->bpt", part_tokens_fp32, positive_text_fp32
        )
        positive_scores = positive_token_scores.max(dim=-1).values.mean(dim=-1)

        negative_text_bank = self._gather_detached(positive_text_fp32)
        negative_target_bank = self._gather_detached(local_targets)
        pair_scores = torch.einsum(
            "bpc,ktc->bkpt", part_tokens_fp32, negative_text_bank
        ).max(dim=-1).values.mean(dim=-1)

        negative_mask = local_targets[:, None].ne(negative_target_bank[None, :])
        valid_rows = negative_mask.any(dim=1)
        masked_scores = pair_scores.masked_fill(
            ~negative_mask, torch.finfo(pair_scores.dtype).min
        )
        hard_negative_scores, hard_negative_indices = masked_scores.max(dim=1)

        per_sample_loss = F.relu(
            self.hnsd_margin - positive_scores + hard_negative_scores
        )
        if valid_rows.any():
            loss = per_sample_loss[valid_rows].mean()
            selected_targets = negative_target_bank[hard_negative_indices[valid_rows]]
            same_class_ratio = (
                selected_targets == local_targets[valid_rows]
            ).float().mean()
            active_ratio = (
                per_sample_loss[valid_rows] > 0
            ).float().mean()
            positive_mean = positive_scores[valid_rows].mean()
            negative_mean = hard_negative_scores[valid_rows].mean()
        else:
            loss = part_tokens_fp32.sum() * 0.0
            zero = loss.detach()
            same_class_ratio = zero
            active_ratio = zero
            positive_mean = zero
            negative_mean = zero

        return {
            "loss": loss,
            "positive_similarity": positive_mean.detach(),
            "negative_similarity": negative_mean.detach(),
            "gap": (positive_mean - negative_mean).detach(),
            "active_ratio": active_ratio.detach(),
            "valid_ratio": valid_rows.float().mean().detach(),
            "same_class_ratio": same_class_ratio.detach(),
        }

    def forward_features(self, x, meta=None, return_aux=False, force_hvp=False, force_hvp_layer=None, hnsd_targets=None):
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

        # Keep the projected local caption tokens only when the training-time
        # HNSD branch is active. Normal training and inference do no extra work.
        use_hnsd = (
            return_aux
            and self.enable_hnsd
            and self.training
            and hnsd_targets is not None
        )
        text_tokens_1 = (
            torch.cat(extra_tokens_1[1:], dim=1)
            if use_hnsd and len(extra_tokens_1) > 1 else None
        )
        text_tokens_2 = (
            torch.cat(extra_tokens_2[1:], dim=1)
            if use_hnsd and len(extra_tokens_2) > 1 else None
        )
        x = self.stage_0(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        for blk in self.stage_1:
            x = blk(x)

        for blk in self.stage_2:
            x = blk(x)

        # V8-Lite 类别语义路由：全局特征 → 类别分布 → 加权类别语义 token
        # 训练/测试完全一致；真实标签只用于 loss_route，不参与 cat_token 生成（无泄漏）
        route_logits = None
        if self.category_bank.numel() > 0:
            z = x.mean(dim=(2, 3))                         # (B, C_feat)，对空间维全局平均
            z = self.route_norm(z)
            route_logits = self.route_head(z)              # (B, num_classes)
            p = torch.softmax(route_logits / self.temperature, dim=-1)
            cat_emb = p.detach() @ self.category_bank      # (B, 768)，detach 让路由头只被 loss_route 训
            cat_1 = self.cat_head_1(cat_emb).unsqueeze(1)  # (B,1,512)
            cat_2 = self.cat_head_2(cat_emb).unsqueeze(1)  # (B,1,1024)
        else:
            cat_1 = self.cls_token_1
            cat_2 = self.cls_token_2
        part_gen_tokens_1 = [cat_1] + extra_tokens_1[1:]
        part_gen_tokens_2 = [cat_2] + extra_tokens_2[1:]

        force_hvp_1 = bool(force_hvp and (force_hvp_layer is None or int(force_hvp_layer) == 1))
        out_1 = self.part_gen_1(x, part_gen_tokens_1, return_aux=return_aux, force_hvp=force_hvp_1)
        if return_aux:
            part_tokens_1, aux_1 = out_1
        else:
            part_tokens_1 = out_1

        extra_tokens_1.append(part_tokens_1)

        H0, W0 = self.img_size // 8, self.img_size // 8
        if self.assess:
            layer_weights1 = []
            layer_weights2 = []
        for ind, blk in enumerate(self.stage_3):
            if ind == 0:
                x, attn = blk(x, H0, W0, extra_tokens_1)
            else:
                x, _ = blk(x, H0, W0)
            if self.assess:
                    layer_weights1.append(attn)
        if not self.only_last_cls:
            cls_1 = x[:, :1, :]
            cls_1 = self.norm_1(cls_1)
            cls_1 = self.cl_1_fc(cls_1)

        x = x[:, self.extra_token_num:, :]

        H1, W1 = self.img_size // 16, self.img_size // 16

        force_hvp_2 = bool(force_hvp and (force_hvp_layer is None or int(force_hvp_layer) == 2))
        out_2 = self.part_gen_2(x, part_gen_tokens_2, return_aux=return_aux, force_hvp=force_hvp_2)
        if return_aux:
            part_tokens_2, aux_2 = out_2
        else:
            part_tokens_2 = out_2

        if return_aux:
            hnsd_zero = part_tokens_2.new_zeros(())
            hnsd_stats = {
                "loss": hnsd_zero,
                "positive_similarity": hnsd_zero.detach(),
                "negative_similarity": hnsd_zero.detach(),
                "gap": hnsd_zero.detach(),
                "active_ratio": hnsd_zero.detach(),
                "valid_ratio": hnsd_zero.detach(),
                "same_class_ratio": hnsd_zero.detach(),
            }
            if (
                use_hnsd
                and text_tokens_1 is not None
                and text_tokens_2 is not None
            ):
                hnsd_1 = self._hnsd_stage_loss(
                    part_tokens_1, text_tokens_1, hnsd_targets
                )
                hnsd_2 = self._hnsd_stage_loss(
                    part_tokens_2, text_tokens_2, hnsd_targets
                )
                hnsd_stats = {
                    key: 0.5 * (hnsd_1[key] + hnsd_2[key])
                    for key in hnsd_stats
                }

        extra_tokens_2.append(part_tokens_2)

        x = x.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        for ind, blk in enumerate(self.stage_4):
            if ind == 0:
                x, attn1 = blk(x, H1, W1, extra_tokens_2)
            else:
                x, _ = blk(x, H1, W1)
            if self.assess:
                    layer_weights2.append(attn1)
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
            part_aux_loss = 0.5 * (aux_1["part_aux_loss"] + aux_2["part_aux_loss"])

            aux_full = {
                # ============================================================
                # Scalar monitoring
                # ============================================================
                "align_loss": align_loss,
                "curv_reg_loss": curv_reg_loss,
                "part_aux_loss": part_aux_loss,
                "s_cls": 0.5 * (aux_1["s_cls"].mean() + aux_2["s_cls"].mean()),
                "curv_entropy": 0.5 * (aux_1["curv_entropy"] + aux_2["curv_entropy"]),
                "curv_weight_max": 0.5 * (aux_1["curv_weight_max"] + aux_2["curv_weight_max"]),
                "curv_weight_mean": 0.5 * (aux_1["curv_weight_mean"] + aux_2["curv_weight_mean"]),
                "hnsd_loss": hnsd_stats["loss"],
                "hnsd_positive_similarity": hnsd_stats["positive_similarity"],
                "hnsd_negative_similarity": hnsd_stats["negative_similarity"],
                "hnsd_gap": hnsd_stats["gap"],
                "hnsd_active_ratio": hnsd_stats["active_ratio"],
                "hnsd_valid_ratio": hnsd_stats["valid_ratio"],
                "hnsd_same_class_ratio": hnsd_stats["same_class_ratio"],

                # ============================================================
                # Category semantics (different resolutions: never add them)
                # ============================================================
                "s_cls_1": aux_1["s_cls"],
                "s_cls_2": aux_2["s_cls"],
                "cls_sem_1": aux_1["cls_sem"],
                "cls_sem_2": aux_2["cls_sem"],

                # ============================================================
                # Student curvature
                # ============================================================
                "student_raw_1": aux_1.get("student_raw", None),
                "student_raw_2": aux_2.get("student_raw", None),
                "student_curvature_1": aux_1.get("student_curvature", aux_1["curvature"]),
                "student_curvature_2": aux_2.get("student_curvature", aux_2["curvature"]),
                "student_curvature_reg_1": aux_1.get("student_curvature_reg", None),
                "student_curvature_reg_2": aux_2.get("student_curvature_reg", None),
                "curvature_1": aux_1["curvature"],
                "curvature_2": aux_2["curvature"],
                "curv_prob_1": aux_1.get("curv_prob", None),
                "curv_prob_2": aux_2.get("curv_prob", None),
                "curv_weight_1": aux_1["curv_weight"],
                "curv_weight_2": aux_2["curv_weight"],

                # ============================================================
                # HVP Teacher
                # In normal eval these are None; force_hvp=True populates them.
                # ============================================================
                "hvp_raw_1": aux_1.get("hvp_raw", None),
                "hvp_raw_2": aux_2.get("hvp_raw", None),
                "hvp_curvature_1": aux_1.get("hvp_curvature", None),
                "hvp_curvature_2": aux_2.get("hvp_curvature", None),

                # ============================================================
                # Semantic grounding
                # ============================================================
                "token_part_sim_1": aux_1.get("token_part_sim", None),
                "token_part_sim_2": aux_2.get("token_part_sim", None),
                "part_assign_1": aux_1["part_assign"],
                "part_assign_2": aux_2["part_assign"],
                "attr_attn_1": aux_1["attr_attn"],
                "attr_attn_2": aux_2["attr_attn"],
                "per_token_sim_1": aux_1["per_token_sim"],
                "per_token_sim_2": aux_2["per_token_sim"],

                # ============================================================
                # Actual attention used to form part tokens
                # attn_raw = softmax(attn_logits) BEFORE attention dropout.
                # This is the correct quantity for Part heatmaps/diversity.
                # ============================================================
                "attn_logits_1": aux_1.get("attn_logits", None),
                "attn_logits_2": aux_2.get("attn_logits", None),
                "attn_raw_1": aux_1.get("attn_raw", None),
                "attn_raw_2": aux_2.get("attn_raw", None),
                "part_attn_1": aux_1.get("part_attn", aux_1.get("attn_raw", None)),
                "part_attn_2": aux_2.get("part_attn", aux_2.get("attn_raw", None)),

                # ============================================================
                # Gate / route
                # ============================================================
                "gate_status_1": aux_1["gate_status"],
                "gate_status_2": aux_2["gate_status"],
                "route_logits": route_logits,
            }
            if self.assess:
                return cls, aux_full, layer_weights1, layer_weights2
            else:
                return cls, aux_full
        else:
            # 不返回 aux_full 的情况（按需保留）
            if self.assess:
                return cls, layer_weights1, layer_weights2
            else:
                return cls
            
    def forward(self, x, meta=None, return_aux=False, force_hvp=False, force_hvp_layer=None, hnsd_targets=None):
        # ---- 原有的 meta 掩码逻辑保持不变 ----
        if meta is not None:
            if self.mask_type == "linear":
                cur_mask_prob = self.mask_prob - self.cur_epoch / self.total_epoch
            else:
                cur_mask_prob = self.mask_prob
            cur_mask_prob = max(0.0, float(cur_mask_prob))
            if cur_mask_prob != 0 and self.training:
                mask = torch.ones_like(meta)
                mask_num = int(meta.size(0) * cur_mask_prob)
                mask_index = torch.randperm(meta.size(0), device=meta.device)[:mask_num]
                mask[mask_index] = 0
                meta = mask * meta

        # ---- 根据 return_aux 和 self.assess 分别调用 forward_features ----
        if return_aux:
            if self.assess:   # 需要返回 aux_full + 权重列表
                feat, aux_full, layer_weights1, layer_weights2 = self.forward_features(x, meta, return_aux=True, force_hvp=force_hvp, force_hvp_layer=force_hvp_layer, hnsd_targets=hnsd_targets)
                return self.head(feat), aux_full, layer_weights1, layer_weights2
            else:             # 只返回 aux_full
                feat, aux_full = self.forward_features(x, meta, return_aux=True, force_hvp=force_hvp, force_hvp_layer=force_hvp_layer, hnsd_targets=hnsd_targets)
                return self.head(feat), aux_full
        else:
            if self.assess:   # 不返回 aux_full，但返回权重列表
                feat, layer_weights1, layer_weights2 = self.forward_features(x, meta, return_aux=False, force_hvp=force_hvp, force_hvp_layer=force_hvp_layer)
                return self.head(feat), layer_weights1, layer_weights2
            else:             # 仅返回分类结果
                feat = self.forward_features(x, meta, return_aux=False, force_hvp=force_hvp, force_hvp_layer=force_hvp_layer)
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
