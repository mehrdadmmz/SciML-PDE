# transformer_3d_aux.py
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from collections import OrderedDict
from timm.layers import trunc_normal_ as __call_trunc_normal_
from timm.layers import drop_path, to_2tuple
from einops import rearrange


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PatchEmbed3D(nn.Module):

    def __init__(
        self,
        img_size=(50, 50, 89),
        patch_size=(10, 10, 9),
        in_chans=4,
        embed_dim=768,
        num_frames=10,
        tubelet_size=2,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.tubelet = tubelet_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # padded spatial grid
        self.grid_size = tuple(
            math.ceil(s / p) * p for s, p in zip(self.img_size, self.patch_size)
        )
        nt = num_frames // tubelet_size
        nx, ny, nz = (g // p for g, p in zip(self.grid_size, self.patch_size))
        self.num_patches = nt * nx * ny * nz

        vox_per_patch = tubelet_size * math.prod(self.patch_size)
        self.proj = nn.Linear(in_chans * vox_per_patch, embed_dim)

    def forward(self, x):  # x: (B,C,T,X,Y,Z)
        B, C, T, X, Y, Z = x.shape
        tt, px, py, pz = self.tubelet, *self.patch_size

        # spatial padding
        pad_z = self.grid_size[2] - Z
        pad_y = self.grid_size[1] - Y
        pad_x = self.grid_size[0] - X

        # reshape → replicate‑pad → reshape back
        x = x.permute(0, 1, 3, 4, 5, 2).contiguous()  # (B,C,X,Y,Z,T)
        x = x.view(B, C * T, X, Y, Z)                 # (B,C·T,X,Y,Z)

        if pad_x or pad_y or pad_z:
            x = F.pad(x, (0, pad_z, 0, pad_y, 0, pad_x), mode="replicate")

        Xp, Yp, Zp = X + pad_x, Y + pad_y, Z + pad_z
        x = x.view(B, C, T, Xp, Yp, Zp)               # (B,C,T,Xp,Yp,Zp)

        # flatten tubelets → tokens
        x = rearrange(
            x,
            "b c (t tt) (x px) (y py) (z pz) -> b (t x y z) (tt px py pz c)",
            tt=tt,
            px=px,
            py=py,
            pz=pz,
        )
        return self.proj(x), (pad_x, pad_y, pad_z)    # tokens, pads


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_vec(pos):
        return [pos / np.power(10000, 2 * (i // 2) / d_hid) for i in range(d_hid)]

    table = np.array([get_vec(p) for p in range(n_position)])
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return torch.tensor(table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"p={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = attn_head_dim or (dim // num_heads)
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        bias = None
        if self.q_bias is not None:
            bias = torch.cat(
                [self.q_bias, torch.zeros_like(
                    self.v_bias, requires_grad=False), self.v_bias]
            )

        qkv = F.linear(x, self.qkv.weight, bias).reshape(
            B, N, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

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
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
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
            drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PretrainVisionTransformerEncoder(nn.Module):
    def __init__(
        self,
        img_size=(64, 64, 178),
        patch_size=(16, 16, 16),
        in_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        tubelet_size=2,
        use_checkpoint=False,
        use_learnable_pos_emb=False,
        num_frames=10,
        dropout_p=0.0,
        patch_embed_cls=PatchEmbed3D,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.dropout_p = dropout_p

        self.patch_embed = patch_embed_cls(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
        n_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = get_sinusoid_encoding_table(n_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    dpr[i],
                    init_values,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x):  # x: (B,C,T,X,Y,Z)
        tokens, pads = self.patch_embed(x)  # (B,N,C_e)

        if tokens.shape[1] != self.pos_embed.shape[1]:
            self.pos_embed = get_sinusoid_encoding_table(tokens.shape[1], self.embed_dim).type_as(
                tokens
            ).to(tokens.device)

        x = tokens + self.pos_embed.type_as(tokens).to(tokens.device).detach()

        for blk in self.blocks:
            if self.dropout_p > 0:
                x = F.dropout(x, p=self.dropout_p)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)
        return x, pads  # latent sequence & pad info


class PretrainVisionTransformerDecoder(nn.Module):
    def __init__(
        self,
        patch_size=(16, 16, 16),
        tubelet_size=2,
        in_chans=4,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=None,
        num_patches=512,
        dropout_p=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.dropout_p = dropout_p

        out_dim = in_chans * tubelet_size * math.prod(patch_size)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                    dpr[i],
                    init_values,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, out_dim)

    def forward(self, x):  # (B,N,C_d)
        for blk in self.blocks:
            if self.dropout_p > 0:
                x = F.dropout(x, p=self.dropout_p)
            x = blk(x)
        x = self.head(self.norm(x))
        return x  # (B,N,out_dim)


class PretrainVisionTransformer(nn.Module):
    """
    Input : (T, B, C, X, Y, Z)
    Output: (B, C, X, Y, Z)   – prediction for timestep T
    """

    def __init__(
        self,
        img_size=(64, 64, 178),
        patch_size=(16, 16, 16),
        tubelet_size=2,
        num_frames=10,
        in_chans=4,
        encoder_embed_dim=768,
        decoder_embed_dim=512,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.in_chans = in_chans

        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            num_frames=num_frames,
            in_chans=in_chans,
            embed_dim=encoder_embed_dim,
            patch_embed_cls=PatchEmbed3D,
        )

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=decoder_embed_dim,
            num_patches=self.encoder.patch_embed.num_patches,
            **{
                k: kwargs[k]
                for k in kwargs
                if k
                in {
                    "depth",
                    "num_heads",
                    "mlp_ratio",
                    "qkv_bias",
                    "qk_scale",
                    "drop_rate",
                    "attn_drop_rate",
                    "drop_path_rate",
                    "norm_layer",
                    "init_values",
                    "dropout_p",
                }
            },
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x):  # x: (T,B,C,X,Y,Z)
        T, B, C, X, Y, Z = x.shape
        assert T == self.num_frames, "initial_step mismatch"

        with torch.no_grad():
            data_std, data_mean = torch.std_mean(
                x, dim=(0, -3, -2, -1), keepdims=True)
            data_std = data_std + 1e-7

        x = (x - data_mean) / data_std

        x = x.permute(1, 2, 0, 3, 4, 5)  # (B,C,T,X,Y,Z)
        tokens, pads = self.encoder(x)
        x = self.encoder_to_decoder(tokens)
        x = self.decoder(x)  # (B,N,out_dim)

        # tokens --> volume
        px, py, pz = self.patch_size
        tt = self.tubelet_size
        n_t = T // tt
        n_x = math.ceil(X / px)
        n_y = math.ceil(Y / py)
        n_z = math.ceil(Z / pz)

        x = rearrange(
            x,
            "b (t x y z) (tt px py pz c) -> b c (t tt) (x px) (y py) (z pz)",
            t=n_t,
            x=n_x,
            y=n_y,
            z=n_z,
            tt=tt,
            px=px,
            py=py,
            pz=pz,
            c=C,
        )

        pad_x, pad_y, pad_z = pads
        if pad_x or pad_y or pad_z:
            x = x[:, :, :, :X, :Y, :Z]

        y_hat = x[:, :, -1] * data_std.squeeze(0) + data_mean.squeeze(0)
        return y_hat  # (B,C,X,Y,Z)


# AUXILIARY wrapper
class Transformer3DAux(nn.Module):

    def __init__(self, **vit_kwargs):
        super().__init__()
        self.vit_core = PretrainVisionTransformer(**vit_kwargs)

    @staticmethod
    def _normalise(x):
        with torch.no_grad():
            data_std, data_mean = torch.std_mean(
                x, dim=(0, -3, -2, -1), keepdims=True)
            data_std = data_std + 1e-7
        x = (x - data_mean) / data_std

        return x, data_std.squeeze(0), data_mean.squeeze(0)

    def forward(self, x, grid, x_aux, grid_aux):  # grids unused
        # primary stream
        # x: [B, T, C, X, Y, Z]  →  (T,B,C,X,Y,Z)
        B, T, C, X, Y, Z = x.shape
        x_p = x.permute(1, 0, 2, 3, 4, 5)
        x_p, std_p, mean_p = self._normalise(x_p)
        y_p_norm = self.vit_core(x_p)                  # (B,C,X,Y,Z)
        y_p = y_p_norm * std_p + mean_p                # de‑norm

        out_primary = y_p.permute(0, 2, 3, 4, 1).unsqueeze(-2)  # B,X,Y,Z,1,C

        # auxiliary stream(s)
        # x_aux: [B,Nₐ,T,C,X,Y,Z]  →  flatten B & Nₐ
        B, N, T, C, X, Y, Z = x_aux.shape
        x_aux_flat = x_aux.reshape(B * N, T, C, X, Y, Z)
        x_a = x_aux_flat.permute(1, 0, 2, 3, 4, 5)
        x_a, std_a, mean_a = self._normalise(x_a)
        y_a_norm = self.vit_core(x_a)                   # (B·N,C,X,Y,Z)
        y_a = y_a_norm * std_a + mean_a

        out_aux = y_a.permute(0, 2, 3, 4, 1).unsqueeze(-2)      # BN,X,Y,Z,1,C

        return out_primary, out_aux


# convenience builder
def build_vmae3d_aux(params):

    vit_kwargs = dict(
        img_size=params.input_size,
        patch_size=params.patch_size,
        in_chans=params.in_chans,
        tubelet_size=params.tubelet_size,
        num_frames=params.n_steps,
        encoder_embed_dim=params.encoder_embed_dim,
        encoder_depth=12,
        encoder_num_heads=params.encoder_num_heads,
        decoder_embed_dim=params.decoder_embed_dim,
        decoder_depth=params.decoder_depth,
        decoder_num_heads=params.decoder_num_heads,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=params.drop_path_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return Transformer3DAux(**vit_kwargs)
