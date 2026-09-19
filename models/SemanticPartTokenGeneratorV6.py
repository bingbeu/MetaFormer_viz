
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_

try:
    from apex import amp
except ImportError:
    amp = None


class SemanticPartTokenGeneratorV6(nn.Module):
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
        curv_reg_weight: float = 0.05,
        curv_weight_max: float = 8.0,
        feat_gain_max: float = 0.25,
        use_category_grounding: bool = True,   # [V6-3] 类别接地开关
        category_scale: float = 0.5,           # [V6] 类别注入强度 γ_cls 的初始生效值
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
        self.curv_tau = max(curv_tau, 0.1)
        self.hvp_probe = hvp_probe
        self.hvp_samples = hvp_samples
        self.curv_norm_eps = curv_norm_eps
        self.curv_reg_weight = curv_reg_weight
        self.curv_weight_max = curv_weight_max
        self.feat_gain_max = feat_gain_max
        self.use_category_grounding = use_category_grounding
        self.scale = embed_dim ** -0.5
        self.eps = 1e-6

        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.class_proj = nn.Linear(embed_dim, embed_dim)    # [V6] 类别语义投影 c
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

        # V3 门控初始值（一字不改，实测该管线门控学不动，靠初始值就位）
        self.curv_feat_alpha = nn.Parameter(torch.tensor(0.1))   # sigmoid(0.1)≈0.525
        self.curv_sem_alpha = nn.Parameter(torch.tensor(0.5))    # tanh(0.5)≈0.462
        self.curv_logit_alpha = nn.Parameter(torch.tensor(1.0))  # tanh(1.0)≈0.762 全强度曲率偏置
        self.sim_logit_alpha = nn.Parameter(torch.tensor(0.5))   # tanh(0.5)≈0.462
        self.attr_logit_alpha = nn.Parameter(torch.tensor(1.0))

        # [V6] 类别接地门控：softplus 恒正。初始值取 softplus⁻¹(category_scale)，
        #      让类别注入从"中等强度"起步（默认 0.5，不强不弱）。
        #      门控学不动是已知事实，所以 category_scale 就是实际生效强度。
        self.gamma_cls = nn.Parameter(
            torch.tensor(math.log(math.exp(float(category_scale)) - 1.0))
        )

        trunc_normal_(self.part_queries, std=0.02)

    # ------------------------------------------------------------------ utils
    def gate_status(self):
        with torch.no_grad():
            return {
                "curv_feat_gain": (self.feat_gain_max * self.curv_feat_alpha.sigmoid()).item(),
                "curv_sem_gate": self.curv_sem_alpha.tanh().item(),
                "curv_logit_gate": self.curv_logit_alpha.tanh().item(),
                "sim_logit_gate": self.sim_logit_alpha.tanh().item(),
                "attr_scale": (0.1 + F.softplus(self.attr_logit_alpha)).item(),
                "gamma_cls": F.softplus(self.gamma_cls).item(),   # [V6] 类别注入强度
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
        # V3 原样：属性接地全强度，不套门控
        part_q = self.part_queries.expand(batch_size, -1, -1)
        sem_k = self.semantic_proj(attr_tokens)
        part_q_norm = F.normalize(part_q, dim=-1)
        sem_k_norm = F.normalize(sem_k, dim=-1)
        attr_logits = part_q_norm @ sem_k_norm.transpose(-2, -1)
        attr_scale = 0.1 + F.softplus(self.attr_logit_alpha)
        attr_attn = torch.softmax(attr_logits * attr_scale, dim=-1)
        sem_per_part = attr_attn @ sem_k
        q = part_q + sem_per_part
        return q, sem_per_part, attr_attn

    def _token_part_similarity(self, x, sem_per_part):
        # V3 原样：token→part
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
        # V3 原样：HVP 老师目标 = per_token_sim.mean()（token→part）
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
        # V3 原样：mean 归一化
        denom = curvature.mean(dim=1, keepdim=True).clamp_min(self.curv_norm_eps).detach()
        return curvature / denom

    # ---------------------------------------------------------------- forward
    def forward(self, x, extra_tokens, return_aux: bool = False, force_hvp: bool = False):
        x = self._flatten_input(x)
        x = self.input_proj(x)
        B, N, C = x.shape

        if not isinstance(extra_tokens, (list, tuple)) or len(extra_tokens) == 0:
            raise ValueError("extra_tokens must be a non-empty list/tuple")

        tokens = [self._expand_token(t, B, x.device, x.dtype) for t in extra_tokens]

        # [V6] tokens[0]=类别语义，tokens[1:]=属性语义（沿用 V3 约定）
        cls_tokens = tokens[0]
        attr_tokens = torch.cat(tokens[1:], dim=1) if len(tokens) > 1 else cls_tokens

        q, sem_per_part, attr_attn = self._part_attribute_semantics(attr_tokens, B)

        q = q.to(dtype=x.dtype)
        sem_per_part = sem_per_part.to(dtype=x.dtype)

        # [V6-1] 类别语义接地进 part query（全局身份约束）
        cls_sem = self.class_proj(cls_tokens).to(dtype=x.dtype)   # (B,1,C)
        if self.use_category_grounding:
            gamma_cls = F.softplus(self.gamma_cls)
            q = q + gamma_cls * cls_sem                            # (B,P,C) 广播

        token_part_sim, per_token_sim, part_assign = \
            self._token_part_similarity(x, sem_per_part)

        # [V6-2] 逐 token 类别相似度（监控/可视化，不参与主损失）
        s_cls = F.cosine_similarity(x, cls_sem.expand_as(x), dim=-1)   # (B,N)

        token_sem = torch.einsum(
            "bnp,bpc->bnc", part_assign.float(), sem_per_part.float()
        ).to(dtype=x.dtype)

        sem_gate = self.curv_sem_alpha.tanh().to(dtype=x.dtype)

        # 分类/特征路径：保持端到端梯度
        forward_input = x + sem_gate * token_sem
        pred_curvature = self.curv_head(forward_input).clamp_min(self.eps)

        curv_reg_loss = pred_curvature.new_zeros(())
        hvp_curvature = None
        student_curvature_reg = None
        teacher_curvature = None

        # 正常训练行为保持不变：force_hvp=False 时仍只在 training + enable_hvp 下计算 HVP。
        # 可视化/评估时可在 model.eval() 下显式 force_hvp=True，额外计算 Teacher，
        # 不改变任何可训练参数，也不改变分类/part-token 前向结果。
        if (
            (self.enable_hvp or force_hvp)
            and (self.training or force_hvp)
            and torch.is_grad_enabled()
        ):
            reg_input = x.detach() + sem_gate * token_sem.detach()
            head_dtype = next(self.curv_head.parameters()).dtype
            reg_input = reg_input.to(dtype=head_dtype)

            def _run_reg_head():
                return self.curv_head(reg_input)

            if amp is not None and hasattr(amp, "disable_casts"):
                try:
                    with amp.disable_casts():
                        pred_curvature_reg = _run_reg_head()
                except AttributeError:
                    # AMP 未初始化（例如可视化脚本直接 build_model 而没经过
                    # amp.initialize）时，_amp_state.handle 不存在，disable_casts
                    # 会抛 AttributeError。此时没有 autocast 生效，直接 fp32 跑即可。
                    pred_curvature_reg = _run_reg_head()
            else:
                try:
                    with torch.amp.autocast("cuda", enabled=False):
                        pred_curvature_reg = _run_reg_head()
                except (TypeError, RuntimeError):
                    pred_curvature_reg = _run_reg_head()

            pred_curvature_reg = pred_curvature_reg.clamp_min(self.eps)

            hvp_curvature = self._compute_hvp_curvature(x, sem_per_part)

            student_curvature_reg = self._normalize_curvature(pred_curvature_reg)
            teacher_curvature = self._normalize_curvature(
                hvp_curvature
            ).detach().to(dtype=student_curvature_reg.dtype)

            # 与原训练完全一致：归一化 Student 与归一化 HVP Teacher 做 Smooth-L1。
            curv_reg_loss = F.smooth_l1_loss(student_curvature_reg, teacher_curvature)

        # 后续使用预测曲率
        curvature = self._normalize_curvature(pred_curvature)
        curv_logits = curvature.squeeze(-1) / self.curv_tau
        curv_prob = torch.softmax(curv_logits, dim=1)                          # (B,N) Σ=1
        curv_weight = (N * curv_prob).clamp(max=self.curv_weight_max)          # (B,N) 有界

        entropy_denom = torch.log(curv_prob.new_tensor(float(max(N, 2))))
        curv_entropy = -(
            curv_prob * curv_prob.clamp_min(self.eps).log()
        ).sum(dim=1).mean() / entropy_denom

        # V3 原样：曲率概率加权对齐损失（非 plain mean）
        align_loss = ((1.0 - per_token_sim) * curv_prob.detach()).sum(dim=1).mean()
        part_aux_loss = align_loss + self.curv_reg_weight * curv_reg_loss

        # 特征增强：中心化 + 有界增益
        feat_gain = self.feat_gain_max * self.curv_feat_alpha.sigmoid()
        centered_weight = curv_weight - curv_weight.mean(dim=1, keepdim=True)
        weighted_x = x * (1.0 + feat_gain * centered_weight.unsqueeze(-1))
        k = self.key_proj(weighted_x)
        v = self.value_proj(weighted_x)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + self.sim_logit_alpha.tanh() * token_part_sim.transpose(1, 2)
        attn_logits = attn_logits + self.curv_logit_alpha.tanh() * torch.log1p(curvature).transpose(1, 2)

        # dropout 前的真实 Part-Token 空间注意力。
        # 这是最适合论文可视化/Part diversity 统计的量；真正的 part token 由其
        # dropout 版本与 v 聚合得到。拆成两个变量不会改变原前向数值。
        attn_raw = torch.softmax(attn_logits, dim=-1)
        attn = self.attn_drop(attn_raw)
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

                # ----------------------------------------------------------
                # Student / HVP Teacher：仅增加观测字段，不改变模型计算。
                # 当前这版训练的蒸馏口径是：
                # normalize(pred_curvature_reg) vs normalize(hvp_curvature)
                # 的 Smooth-L1，而不是 zscore/log1p 版本。
                # ----------------------------------------------------------
                "student_raw": pred_curvature.detach(),                 # (B,N,1)
                "student_curvature": curvature.detach(),                # (B,N,1), mean-normalized
                "student_curvature_reg": (
                    None if student_curvature_reg is None
                    else student_curvature_reg.detach()
                ),
                "curvature": curvature.detach(),
                "hvp_raw": None if hvp_curvature is None else hvp_curvature.detach(),
                "hvp_curvature": (
                    None if teacher_curvature is None
                    else teacher_curvature.detach()
                ),

                # Curvature policy
                "curv_prob": curv_prob.detach(),
                "curv_weight": curv_weight.detach(),
                "curv_weight_max": curv_weight.max().detach(),
                "curv_weight_mean": curv_weight.mean().detach(),
                "curv_entropy": curv_entropy.detach(),

                # Semantic grounding
                "attr_attn": attr_attn.detach(),
                "token_part_sim": token_part_sim.detach(),
                "part_assign": part_assign.detach(),
                "per_token_sim": per_token_sim.detach(),
                "s_cls": s_cls.detach(),              # [V6]
                "cls_sem": cls_sem.detach(),          # [V6]

                # 最终 Part Token 的真实空间注意力（dropout 前，推荐用于可视化）
                "attn_logits": attn_logits.detach(),
                "attn_raw": attn_raw.detach(),
                "part_attn": attn_raw.detach(),       # 语义清晰的别名

                "gate_status": self.gate_status(),
            }
        return part_tokens


# 若训练脚本按旧名导入，取消下一行注释即可无缝替换：
# SemanticPartTokenGenerator = SemanticPartTokenGeneratorV6
