class SemanticPartTokenGenerator1(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        num_parts,
        attn_drop=0.0,
        enable_hvp=True,
        assign_scale=5.0,
        curv_tau=1.0,
        hvp_probe="rademacher",
        hvp_samples=4,
        curv_norm_eps=1e-4,
        curv_reg_weight=0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_parts = num_parts
        self.enable_hvp = enable_hvp
        self.assign_scale = assign_scale
        self.curv_tau = curv_tau
        self.hvp_probe = hvp_probe
        self.hvp_samples = hvp_samples
        self.curv_norm_eps = curv_norm_eps
        self.curv_reg_weight = curv_reg_weight
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

        self.curv_feat_alpha = nn.Parameter(torch.tensor(0.1))
        self.curv_logit_alpha = nn.Parameter(torch.tensor(0.1))
        self.sim_logit_alpha = nn.Parameter(torch.tensor(0.1))
        self.attr_logit_alpha = nn.Parameter(torch.tensor(1.0))

        trunc_normal_(self.part_queries, std=0.02)

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
            raise ValueError(f"Expected x to be 3D or 4D, got {tuple(x.shape)}")
        return x

    def _part_attribute_semantics(self, attr_tokens, batch_size):
        part_q = self.part_queries.expand(batch_size, -1, -1)
        sem_k = self.semantic_proj(attr_tokens)

        part_q_norm = F.normalize(part_q, dim=-1)
        sem_k_norm = F.normalize(sem_k, dim=-1)
        attr_logits = part_q_norm @ sem_k_norm.transpose(-2, -1)
        attr_attn = torch.softmax(attr_logits * self.attr_logit_alpha.clamp(0.1, 20.0), dim=-1)

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
        raise ValueError("hvp_probe must be 'rademacher' or 'normal' for Hutchinson diagonal estimation")

    def _compute_hvp_curvature(self, x, sem_per_part):
        with torch.enable_grad():
            hvp_x = x.detach().float().requires_grad_(True)
            hvp_sem = sem_per_part.detach().float()
            _, per_token_sim, _ = self._token_part_similarity(hvp_x, hvp_sem)

            obj = per_token_sim.mean()
            grad = torch.autograd.grad(obj, hvp_x, create_graph=True)[0]

            diag_acc = torch.zeros_like(hvp_x)
            sample_count = max(1, int(self.hvp_samples))
            for sample_idx in range(sample_count):
                probe = self._make_probe(hvp_x)
                retain_graph = sample_idx < sample_count - 1
                hvp = torch.autograd.grad((grad * probe).sum(), hvp_x, retain_graph=retain_graph)[0]
                diag_acc = diag_acc + probe * hvp

            diag_h = diag_acc / float(sample_count)
            curvature = diag_h.norm(p=2, dim=-1, keepdim=True).detach()

        return curvature.to(dtype=x.dtype, device=x.device)

    def _normalize_curvature(self, curvature):
        denom = curvature.amax(dim=1, keepdim=True).clamp_min(self.curv_norm_eps)
        return curvature / denom

    def forward(self, x, extra_tokens, return_aux=False):
        x = self._flatten_input(x)
        x = self.input_proj(x)
        B, N, C = x.shape

        tokens = [self._expand_token(t, B, x.device, x.dtype) for t in extra_tokens]
        # token[0] is the class token; attribute/text tokens follow it.
        attr_tokens = torch.cat(tokens[1:], dim=1) if len(tokens) > 1 else tokens[0]

        q, sem_per_part, attr_attn = self._part_attribute_semantics(attr_tokens, B)
        token_part_sim, per_token_sim, part_assign = self._token_part_similarity(x, sem_per_part)

        # pred_curvature = self.curv_head(x).clamp_min(self.eps)
        sem_summary = sem_per_part.mean(dim=1, keepdim=True).expand_as(x)   # (B, N, C)
        curv_input = x + sem_summary                                        # (B, N, C)
        pred_curvature = self.curv_head(curv_input).clamp_min(self.eps)
        curv_reg_loss = pred_curvature.new_zeros(())
        hvp_curvature = None
        if self.enable_hvp and self.training:
            hvp_curvature = self._compute_hvp_curvature(x, sem_per_part)
            curv_reg_loss = F.smooth_l1_loss(
                self._normalize_curvature(pred_curvature),
                self._normalize_curvature(hvp_curvature).detach(),
            )

        curvature = self._normalize_curvature(pred_curvature)
        curv_logits = curvature.squeeze(-1) / self.curv_tau
        curv_weight = N * torch.softmax(curv_logits, dim=1)

        align_loss = ((1.0 - per_token_sim) * curv_weight.detach()).mean(dim=1).mean()
        part_aux_loss = align_loss + self.curv_reg_weight * curv_reg_loss

        weighted_x = x * (1.0 + self.curv_feat_alpha.sigmoid() * curv_weight.unsqueeze(-1))
        k = self.key_proj(weighted_x)
        v = self.value_proj(weighted_x)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + self.sim_logit_alpha.tanh() * token_part_sim.transpose(1, 2)
        attn_logits = attn_logits + self.curv_logit_alpha.tanh() * torch.log1p(curvature).transpose(1, 2)

        attn = self.attn_drop(torch.softmax(attn_logits, dim=-1))
        part_tokens = attn @ v
        part_tokens = self.out_norm(self.out_proj(part_tokens) + self.part_queries.expand(B, -1, -1))

        if return_aux:
            return part_tokens, {
                "align_loss": align_loss,
                "curv_reg_loss": curv_reg_loss,
                "part_aux_loss": part_aux_loss,
                "curvature": curvature.detach(),
                "hvp_curvature": None if hvp_curvature is None else self._normalize_curvature(hvp_curvature).detach(),
                "curv_weight": curv_weight.detach(),
                "attr_attn": attr_attn.detach(),
                "part_assign": part_assign.detach(),
                "per_token_sim": per_token_sim.detach(),
            }
        return part_tokens
