# transformer_rd.py

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import to_2tuple
from timm.layers import trunc_normal_ as _trunc_normal_
from timm.layers import drop_path
from functools import partial
from einops import rearrange


from timm.layers import trunc_normal_ as _trunc_normal


def trunc_normal_(tensor, mean=0., std=1.):
    _trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(
                self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(n_position, d_hid):

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i)
                              for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


# Patch embedding & encoder / decoder


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size,
                                           patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # BCTHW -> BC'T'H'W' -> BC'(T'H'W')
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=512, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 use_learnable_pos_emb=False, num_frames=16, dropout_p=0.):
        super().__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, tubelet_size=tubelet_size, num_frames=num_frames)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint
        self.dropout_p = dropout_p

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask=None):
        _, _, T, _, _ = x.shape  # B, C, T, H, W
        x = self.patch_embed(x)

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        B, _, C = x.shape
        if mask is not None:
            x_vis = x[~mask].reshape(B, -1, C)  # ~mask means visible
        else:
            x_vis = x.reshape(B, -1, C)

        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:
            for blk in self.blocks:
                if self.dropout_p > 0:
                    x_vis = F.dropout(x_vis, p=self.dropout_p)
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask=None):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x


class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False,
                 num_classes_ssl=0, dropout_p=0.):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == encoder.in_chans * tubelet_size * patch_size ** 2
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.dropout_p = dropout_p

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.num_classes_ssl = num_classes_ssl
        if num_classes_ssl > 0:
            self.head_ssl = nn.Linear(embed_dim, num_classes_ssl)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num=-1, return_feature=False):
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
            # only return the mask tokens predict pixels
            x = self.head_ssl(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))

        if return_feature:
            return x, feature
        else:
            return x


# Vision-Transformer backbone


class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=512,
                 patch_size=16,
                 in_chans=3,
                 encoder_num_classes=0,
                 encoder_embed_dim=768,
                 encoder_depth=12,
                 encoder_num_heads=12,
                 decoder_num_classes=1536,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 tubelet_size=2,
                 num_classes=0,  # avoid the error from create_fn in timm
                 num_frames=16,
                 ssl=False,
                 dropout_p=0.,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.ssl = ssl
        self.dropout_p = dropout_p
        self.decoder_embed_dim = decoder_embed_dim
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
            dropout_p=dropout_p,)

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
            num_classes_ssl=tubelet_size * in_chans * patch_size ** 2 if ssl else 0,
            dropout_p=dropout_p,)
        # num_classes_ssl=num_frames * tubelet_size * patch_size ** 2 if ssl else 0)

        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False)
        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder.patch_embed.num_patches, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        trunc_normal_(self.mask_token, std=.02)

    def set_dropout_p(self, p):
        self.dropout_p = p
        self.encoder.dropout_p = p
        self.decoder.dropout_p = p

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, mask=None, return_feature=False):
        '''
        x: T, B, C, H, W
        '''
        T_in, B, C_in, H, W = x.shape

        with torch.no_grad():
            data_std, data_mean = torch.std_mean(
                x, dim=(0, -2, -1), keepdims=True)  # T, H, W
            data_std = data_std + 1e-7  # Orig 1e-7
        x = (x - data_mean) / (data_std)

        x = x.permute(1, 2, 0, 3, 4)
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask)  # [B, N_vis, C_e]
        if self.dropout_p > 0:
            x_vis = F.dropout(x_vis, p=self.dropout_p)
        x_vis = self.encoder_to_decoder(x_vis)  # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order,
        # but shuffle the pos embedding accorddingly.
        if mask is not None:
            expand_pos_embed = self.pos_embed.expand(
                B, -1, -1).type_as(x).to(x.device).clone().detach()
            pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
            if mask.sum() == 0:
                # mask_ratio = 0: all tokens are visible
                x_full = x_vis + pos_emd_vis  # [B, N, C_d]
                x = self.decoder(x_full, 0)  # [B, :, 3 * 16 * 16]
                x = rearrange(x, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=T_in//self.tubelet_size, h=H //
                              self.patch_size, w=W//self.patch_size, p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size, c=C_in)
                x = x.permute(2, 0, 1, 3, 4)
                # TODO TBCHW: All state labels in the batch should be identical
                x = x * data_std + data_mean
                x = x.permute(1, 2, 0, 3, 4)
                x = rearrange(x, 'b c (t p0) (h p1) (w p2) -> b t h w (p0 p1 p2 c)', t=T_in//self.tubelet_size, h=H //
                              self.patch_size, w=W//self.patch_size, p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size, c=C_in)
            else:
                pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
                x_full = torch.cat(
                    # [B, N, C_d]
                    [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
                # [B, N_mask, 3 * 16 * 16]
                x = self.decoder(x_full, pos_emd_mask.shape[1])

                x = x * data_std + data_mean
            return x
        else:
            if return_feature:
                x, feature = self.decoder(x_vis, return_feature=True)
            else:
                feature = None
                x = self.decoder(x_vis)
            x = rearrange(x, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=T_in//self.tubelet_size, h=H //
                          self.patch_size, w=W//self.patch_size, p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size, c=C_in)
            x = x.permute(2, 0, 1, 3, 4)
            x = x * data_std + data_mean  # All state labels in the batch should be identical
            if feature:
                feature = rearrange(feature, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=T_in//self.tubelet_size, h=H//self.patch_size,
                                    w=W//self.patch_size, p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size, c=self.decoder_embed_dim)
                feature = feature.permute(2, 0, 1, 3, 4)
                return x[-1], feature[-1]
            else:
                return x[-1]


class ViT2d(nn.Module):

    def __init__(
        self,
        num_channels: int,
        img_size: int | tuple[int, int] = 256,
        patch_size: int = 16,
        initial_step: int = 10,
        tubelet_size: int = 2,
        **vit_kwargs,
    ):
        super().__init__()
        # Vision-Transformer backbone
        self.initial_step = initial_step
        self.num_channels = num_channels
        self.tubelet_size = tubelet_size

        self.vit = PretrainVisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=num_channels,
            num_frames=initial_step,
            tubelet_size=tubelet_size,
            **vit_kwargs,
        )

    # forward – robust to input layout, returns (B, C, H, W)
    def forward(self, x: torch.Tensor, grid: torch.Tensor | None = None):

        C = self.num_channels
        T = self.initial_step

        # time-first --> batch-first
        if x.ndim == 5 and x.shape[0] == T:
            x = x.permute(1, 0, 2, 3, 4)          # (B,T,C,H,W)

        # batch-first --> flattened
        if x.ndim == 5 and x.shape[1] == T:
            B, _, _, H, W = x.shape
            x = (x.permute(0, 3, 4, 1, 2)         # (B,H,W,T,C)
                 .contiguous()
                 .view(B, H, W, T * C))         # (B,H,W,T·C)

        # now expect flattened
        assert x.ndim == 4, f"Unexpected tensor rank: {x.shape}"
        B, H, W, TC = x.shape
        assert TC == T * C, \
            f"Expected last dim {T*C}, got {TC}"

        # reshape for ViT --> (T,B,C,H,W)
        frames = (x.view(B, H, W, T, C)
                  .permute(3, 0, 4, 1, 2)       # (T,B,C,H,W)
                  .contiguous())

        # Vision Transformer
        pred = self.vit(frames)                   # (B,C,H,W)

        return pred


# Helper – build_vit2d()
def build_vit2d(cfg):

    return ViT2d(
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
