from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError("Install timm before running ViT experiments.") from exc

from models.SemanticPartTokenGeneratorV6 import SemanticPartTokenGeneratorV6


def _load_category_bank(path, num_classes, semantic_dim):
    if path is None:
        bank = torch.empty(num_classes, semantic_dim)
        nn.init.trunc_normal_(bank, std=0.02)
        return bank, True
    path = Path(path).expanduser()
    bank = torch.from_numpy(np.load(path)).float()
    expected = (num_classes, semantic_dim)
    if tuple(bank.shape) != expected:
        raise ValueError(f"Expected category bank {expected}, got {tuple(bank.shape)} from {path}")
    return bank, False


def _make_vit(backbone, pretrained, num_classes, drop_path):
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path,
    )


class ViTBaseline(nn.Module):
    def __init__(self, backbone, pretrained, num_classes, drop_path):
        super().__init__()
        self.backbone = _make_vit(backbone, pretrained, num_classes, drop_path)

    def forward(self, images, semantics=None, return_aux=False):
        logits = self.backbone(images)
        if return_aux:
            return logits, {}
        return logits

    def backbone_parameters(self):
        return (
            parameter
            for name, parameter in self.backbone.named_parameters()
            if not name.startswith("head.")
        )


class CurvPartViT(nn.Module):
    """ViT with training-only curvature teachers and first-order inference."""

    def __init__(
        self,
        backbone,
        pretrained,
        num_classes,
        insert_layers=(8, 10),
        num_parts=8,
        semantic_dim=768,
        category_bank_path=None,
        drop_path=0.1,
        hvp_samples=4,
        curv_tau=1.0,
        curv_reg_weight=0.1,
    ):
        super().__init__()
        self.backbone = _make_vit(backbone, pretrained, num_classes, drop_path)
        if not hasattr(self.backbone, "blocks") or not hasattr(self.backbone, "patch_embed"):
            raise TypeError(f"{backbone} is not a standard timm VisionTransformer")
        if getattr(self.backbone, "dist_token", None) is not None:
            raise ValueError("Distilled ViT variants are not supported; choose a standard ViT.")

        depth = len(self.backbone.blocks)
        self.insert_layers = tuple(sorted(set(int(i) for i in insert_layers)))
        if not self.insert_layers or self.insert_layers[0] < 0 or self.insert_layers[-1] >= depth:
            raise ValueError(f"insert_layers must be within [0, {depth - 1}]")

        self.embed_dim = int(self.backbone.embed_dim)
        self.num_patches = int(self.backbone.patch_embed.num_patches)
        self.semantic_proj = nn.Linear(semantic_dim, self.embed_dim)
        self.route_norm = nn.LayerNorm(self.embed_dim)
        self.route_head = nn.Linear(self.embed_dim, num_classes)

        bank, learnable = _load_category_bank(category_bank_path, num_classes, semantic_dim)
        if learnable:
            self.category_bank = nn.Parameter(bank)
        else:
            self.register_buffer("category_bank", bank, persistent=True)

        self.generators = nn.ModuleDict()
        for layer in self.insert_layers:
            self.generators[str(layer)] = SemanticPartTokenGeneratorV6(
                in_dim=self.embed_dim,
                embed_dim=self.embed_dim,
                num_parts=num_parts,
                enable_hvp=True,
                hvp_samples=hvp_samples,
                curv_tau=curv_tau,
                curv_reg_weight=curv_reg_weight,
            )

    def backbone_parameters(self):
        return (
            parameter
            for name, parameter in self.backbone.named_parameters()
            if not name.startswith("head.")
        )

    def _embed_images(self, images):
        x = self.backbone.patch_embed(images)
        if x.ndim == 4:
            x = x.flatten(2).transpose(1, 2)
        cls = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        if x.shape[1] != self.backbone.pos_embed.shape[1]:
            raise ValueError(
                f"Token count {x.shape[1]} does not match positional embedding "
                f"{self.backbone.pos_embed.shape[1]}; use the backbone's configured input size."
            )
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        patch_drop = getattr(self.backbone, "patch_drop", None)
        if patch_drop is not None:
            x = patch_drop(x)
        norm_pre = getattr(self.backbone, "norm_pre", None)
        if norm_pre is not None:
            x = norm_pre(x)
        return x

    @staticmethod
    def _mean_aux(aux_list):
        keys = ("align_loss", "curv_reg_loss", "part_aux_loss")
        output = {}
        for key in keys:
            output[key] = torch.stack([aux[key] for aux in aux_list]).mean()
        return output

    def forward(self, images, semantics=None, return_aux=False):
        x = self._embed_images(images)
        aux_list, route_logits_list = [], []
        projected_semantics = None
        if semantics is not None and semantics.shape[1] > 0:
            projected_semantics = self.semantic_proj(semantics.to(dtype=x.dtype))

        for index, block in enumerate(self.backbone.blocks):
            if index in self.insert_layers:
                # Only original patch tokens teach/generate parts. Earlier parts remain in
                # the main sequence and continue to interact through later ViT blocks.
                patch_tokens = x[:, 1 : 1 + self.num_patches]
                route_logits = self.route_head(self.route_norm(x[:, 0]))
                route_prob = torch.softmax(route_logits, dim=-1)
                category = route_prob.detach() @ self.category_bank.to(dtype=x.dtype)
                category = self.semantic_proj(category).unsqueeze(1)
                conditions = [category]
                if projected_semantics is not None:
                    conditions.append(projected_semantics)
                generated = self.generators[str(index)](
                    patch_tokens, conditions, return_aux=return_aux
                )
                if return_aux:
                    part_tokens, part_aux = generated
                    aux_list.append(part_aux)
                else:
                    part_tokens = generated
                route_logits_list.append(route_logits)
                x = torch.cat((x, part_tokens), dim=1)
            x = block(x)

        x = self.backbone.norm(x)
        cls = x[:, 0]
        fc_norm = getattr(self.backbone, "fc_norm", None)
        if fc_norm is not None:
            cls = fc_norm(cls)
        pre_logits = getattr(self.backbone, "pre_logits", None)
        if pre_logits is not None:
            cls = pre_logits(cls)
        head_drop = getattr(self.backbone, "head_drop", None)
        if head_drop is not None:
            cls = head_drop(cls)
        logits = self.backbone.head(cls)
        if not return_aux:
            return logits

        aux = self._mean_aux(aux_list)
        aux["route_logits"] = torch.stack(route_logits_list).mean(dim=0)
        return logits, aux


def build_model(args):
    if args.model == "vit":
        return ViTBaseline(
            args.backbone, args.pretrained, args.num_classes, args.drop_path
        )
    if args.model == "curvpart_vit":
        return CurvPartViT(
            backbone=args.backbone,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            insert_layers=args.insert_layers,
            num_parts=args.num_parts,
            semantic_dim=args.semantic_dim,
            category_bank_path=args.category_bank,
            drop_path=args.drop_path,
            hvp_samples=args.hvp_samples,
            curv_tau=args.curv_tau,
            curv_reg_weight=args.curv_reg_weight,
        )
    raise ValueError(f"Unknown model: {args.model}")
