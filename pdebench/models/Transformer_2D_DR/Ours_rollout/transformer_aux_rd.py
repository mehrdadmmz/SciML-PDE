from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
import numpy as np
from timm.layers import drop_path, to_2tuple, trunc_normal_ as _trunc_normal_


def trunc_normal_(tensor, mean: float = 0.0, std: float = 1.0):
    """Truncated normal initialiser (wrapper around timm's version)."""
    _trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class DropPath(nn.Module):
    """Stochastic depth per sample."""

    def __init__(self, drop_prob: float | None = None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:  # noqa: D401
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_head_dim: int | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        all_head_dim = head_dim * num_heads
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        B, N, C = x.shape
        bias = None
        if self.q_bias is not None:
            bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias).reshape(
            B, N, 3, self.num_heads, -1)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # type: ignore

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        init_values: float | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn_head_dim: int | None = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values and init_values > 0.0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Sinusoidal positional‑encoding lookup table."""

    def get_pos_angle_vec(pos: int) -> list[float]:
        return [pos / (10000 ** (2 * (j // 2) / d_hid)) for j in range(d_hid)]

    table = np.array([get_pos_angle_vec(i) for i in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float32).unsqueeze(0)


class PatchEmbed(nn.Module):
    """Tubelet (T×H×W) → patch embedding."""

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        num_frames: int = 16,
        tubelet_size: int = 2,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)

        num_patches = (
            (img_size[0] // patch_size[0])
            * (img_size[1] // patch_size[1])
            * (num_frames // self.tubelet_size)
        )
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # B C T H W → B N C
        B, C, T, H, W = x.shape
        assert (H, W) == self.img_size, "Input size mismatch with img_size."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size: int | Tuple[int, int] = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_values: float | None = None,
        tubelet_size: int = 2,
        use_checkpoint: bool = False,
        use_learnable_pos_emb: bool = False,
        num_frames: int = 16,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, tubelet_size
        )
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.dropout_p = dropout_p

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed.to(x.device)
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                if self.dropout_p > 0:
                    x = F.dropout(x, p=self.dropout_p)
                x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        num_classes: int = 768,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_values: float | None = None,
        num_patches: int = 196,
        tubelet_size: int = 2,
        use_checkpoint: bool = False,
        num_classes_ssl: int = 0,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.dropout_p = dropout_p

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.num_classes_ssl = num_classes_ssl
        if num_classes_ssl > 0:
            self.head_ssl = nn.Linear(embed_dim, num_classes_ssl)

    def forward(self, x: torch.Tensor, return_token_num: int = -1, return_feature: bool = False):
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for blk in self.blocks:
                if self.dropout_p > 0:
                    x = F.dropout(x, p=self.dropout_p)
                x = blk(x)
        if self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p)
        feature = x
        if return_token_num >= 0:
            x = self.head_ssl(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return (x, feature) if return_feature else x


class PretrainVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int | Tuple[int, int] = 512,
        patch_size: int = 16,
        in_chans: int = 3,
        encoder_num_classes: int = 0,
        encoder_embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        decoder_num_classes: int = 1536,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        init_values: float = 0.0,
        use_learnable_pos_emb: bool = False,
        use_checkpoint: bool = False,
        tubelet_size: int = 2,
        num_classes: int = 0,  # dummy for timm create_fn
        num_frames: int = 16,
        ssl: bool = False,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.decoder_embed_dim = decoder_embed_dim
        self.ssl = ssl
        self.dropout_p = dropout_p

        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            use_learnable_pos_emb=use_learnable_pos_emb,
            num_frames=num_frames,
            dropout_p=dropout_p,
        )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            num_classes_ssl=tubelet_size * in_chans * patch_size**2 if ssl else 0,
            dropout_p=dropout_p,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False)
        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, return_feature: bool = False):
        T_in, B, C_in, H, W = x.shape  # T B C H W
        with torch.no_grad():
            data_std, data_mean = torch.std_mean(
                x, dim=(0, -2, -1), keepdim=True)
            data_std = data_std + 1e-7
        x = (x - data_mean) / data_std

        x = x.permute(1, 2, 0, 3, 4)  # B C T H W
        feat = self.encoder(x)  # B N C_e
        if self.dropout_p > 0:
            feat = F.dropout(feat, p=self.dropout_p)
        feat = self.encoder_to_decoder(feat)  # B N C_d

        if mask is not None:
            raise NotImplementedError("Masked pretraining path not used here.")
        pred = self.decoder(feat)  # B N (patch_vec)
        pred = rearrange(
            pred,
            "b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)",
            t=T_in // self.tubelet_size,
            h=H // self.patch_size,
            w=W // self.patch_size,
            p0=self.tubelet_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=C_in,
        )
        pred = pred.permute(2, 0, 1, 3, 4)  # T B C H W
        pred = pred * data_std + data_mean
        return pred[-1]  # last timestep (B C H W)


def _flatten_context(x: torch.Tensor, T: int, C: int) -> Tuple[torch.Tensor, int, int, int]:
    if x.ndim == 5 and x.shape[0] == T:
        x = x.permute(1, 0, 2, 3, 4)  # B T C H W
    if x.ndim == 5 and x.shape[1] == T:
        B, _, C_, H, W = x.shape
        assert C_ == C
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(B, H, W, T * C)
    assert x.ndim == 4, f"Unexpected shape {x.shape}"
    B, H, W, TC = x.shape
    assert TC == T * C, f"Expected {T*C}, got {TC}"
    return x, B, H, W


class ViT2dAux(nn.Module):
    def __init__(
        self,
        num_channels: int,
        img_size: int | Tuple[int, int] = 256,
        patch_size: int = 16,
        initial_step: int = 10,
        tubelet_size: int = 2,
        **vit_kwargs: Any,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.initial_step = initial_step
        self.vit = PretrainVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_channels,
            num_frames=initial_step,
            tubelet_size=tubelet_size,
            **vit_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        grid: torch.Tensor | None,
        x_aux: torch.Tensor,
        grid_aux: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        C, T = self.num_channels, self.initial_step

        x_f, B, H, W = _flatten_context(x, T, C)          # (B,H,W,T·C)
        frames = x_f.view(B, H, W, T, C)                  # (B,H,W,T,C)
        frames = frames.permute(3, 0, 4, 1, 2).contiguous()   # (T,B,C,H,W)
        out_p = self.vit(frames)                          # (B,C,H,W)

        x_af, B_aux, H_aux, W_aux = _flatten_context(x_aux, T, C)
        frames_aux = x_af.view(B_aux, H_aux, W_aux, T, C)
        frames_aux = frames_aux.permute(3, 0, 4, 1, 2).contiguous()
        out_a = self.vit(frames_aux)                      # (B_aux,C,H,W)

        return out_p, out_a


def build_vit2d_aux(cfg: SimpleNamespace) -> ViT2dAux:
    return ViT2dAux(
        num_channels=cfg.in_chans,
        img_size=cfg.input_size,
        patch_size=cfg.patch_size,
        initial_step=cfg.n_steps,
        tubelet_size=cfg.tubelet_size,
        encoder_embed_dim=cfg.encoder_embed_dim,
        encoder_depth=cfg.encoder_depth,
        decoder_depth=cfg.decoder_depth,
        encoder_num_heads=cfg.encoder_num_heads,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_num_heads=cfg.decoder_num_heads,
        drop_path_rate=cfg.drop_path_rate,
        ssl=cfg.ssl,
    )
