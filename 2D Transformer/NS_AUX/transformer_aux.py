# transformer_aux.py
# extends transformer.py to support an auxiliary data stream
# please refer to the original transformer.py for more details
import torch
import torch.nn.functional as F
from einops import rearrange

# reuse some imports from transformer.py
from transformer import (
    PretrainVisionTransformer,
    get_sinusoid_encoding_table,      # needed for build helper
    _cfg                              # timm-style config helper
)


class PretrainVisionTransformerAux(PretrainVisionTransformer):
    """
    Vision Transformer that produces two parallel predictions:
    primary   : from x                (T, B, C, H, W)
    auxiliary : from x_aux            (T_aux, B_aux, C, H, W)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # single decoder head will be replace with two task-specific heads: primary and auxiliary
        # primary head is the same as before
        decoder_dim = self.decoder.embed_dim
        self.head_primary = torch.nn.Linear(
            decoder_dim, self.decoder.num_classes)
        self.head_auxiliary = torch.nn.Linear(
            decoder_dim, self.decoder.num_classes)

    # helpers
    def _encode(self, x_norm, dropout=True):
        """
        Shared encoder --> proj --> decoder-token sequence
        """
        vis = self.encoder(x_norm)                         # [B, N_vis, C_e]
        if self.dropout_p > 0 and dropout:
            vis = F.dropout(vis, p=self.dropout_p)
        vis = self.encoder_to_decoder(vis)                 # [B, N_vis, C_d]
        return vis

    def _decode_and_reshape(self, tokens, T_in, C_in, H, W):
        """
        Shared decoder (without head) + reshape back to video volume
        """
        x = self.decoder(tokens)  # [B, N, patch*tube*...*C_in]
        # tokens -->  volume (see original transformer.py)
        x = rearrange(
            x,
            'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)',
            t=T_in // self.tubelet_size,
            h=H // self.patch_size,
            w=W // self.patch_size,
            p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size, c=C_in
        )
        return x.permute(2, 0, 1, 3, 4)                    # (T, B, C, H, W)

    # forward
    def forward(self, x, x_aux, *, mask=None):
        """
        Inputs
        x      : primary   tensor (T,  B, C, H, W)
        x_aux  : auxiliary tensor (T', B', C, H, W)

        Returns
        out_primary   : [B,  C, H, W] – last predicted frame from x
        out_auxiliary : [B', C, H, W] – last predicted frame from x_aux
        """

        # x      : primary   tensor (T,  B, C, H, W)
        # x_aux  : auxiliary tensor (T', B', C, H, W)
        T,  B,  C, H, W = x.shape
        T2, B2, _, _, _ = x_aux.shape

        # Normalise both streams independently
        with torch.no_grad():
            std_p, mean_p = torch.std_mean(x, dim=(0, -2, -1), keepdims=True)
            std_a, mean_a = torch.std_mean(
                x_aux, dim=(0, -2, -1), keepdims=True)
            std_p = std_p + 1e-7
            std_a = std_a + 1e-7

        x_norm = (x - mean_p) / std_p
        x_aux_norm = (x_aux - mean_a) / std_a

        # Rearrange to (B, C, T, H, W) as expected by encoder
        x_norm = x_norm.permute(1, 2, 0, 3, 4)      # B, C, T, H, W
        x_aux_norm = x_aux_norm.permute(1, 2, 0, 3, 4)  # B2, C, T2, H, W

        # Shared encoder / decoder
        tokens_p = self._encode(x_norm)
        tokens_a = self._encode(x_aux_norm)

        vol_p = self._decode_and_reshape(tokens_p, T,  C, H, W)
        vol_a = self._decode_and_reshape(tokens_a, T2, C, H, W)

        # heads for both streams
        # last temporal slice
        last_p = vol_p[-1]          # [B,  C, H, W]
        last_a = vol_a[-1]          # [B2, C, H, W]

        # Flatten spatial dims for per-pixel head
        Bflat, Cflat = last_p.shape[0], last_p.shape[1]
        out_p = self.head_primary(last_p.view(Bflat, Cflat, -1).transpose(1, 2)
                                  ).transpose(1, 2).view_as(last_p)
        out_a = self.head_auxiliary(last_a.view(last_a.shape[0], Cflat, -1).transpose(1, 2)
                                    ).transpose(1, 2).view_as(last_a)

        # de-normalise (multiply by its own std/mean) & return
        out_p = out_p * std_p.squeeze(-1).squeeze(-1) + \
            mean_p.squeeze(-1).squeeze(-1)
        out_a = out_a * std_a.squeeze(-1).squeeze(-1) + \
            mean_a.squeeze(-1).squeeze(-1)

        # return predictions for both streams
        # out_p : [B,  C, H, W] – last predicted frame from x
        # out_a : [B', C, H, W] – last predicted frame from x_aux
        return out_p, out_a


# Helper builder (mirrors build_vmae from transformer.py)
def build_vmae_aux(params):
    """
    Instantiate PretrainVisionTransformerAux from the same Params object
    used by build_vmae in transformer.py
    """
    model = PretrainVisionTransformerAux(
        img_size=params.input_size,
        patch_size=params.patch_size,
        in_chans=params.in_chans,
        encoder_embed_dim=params.encoder_embed_dim,
        encoder_depth=12,
        decoder_depth=params.decoder_depth,
        encoder_num_heads=params.encoder_num_heads,
        mlp_ratio=4, qkv_bias=True,
        encoder_num_classes=0,
        decoder_num_classes=params.decoder_num_classes,
        tubelet_size=params.tubelet_size,
        decoder_embed_dim=params.decoder_embed_dim,
        decoder_num_heads=params.decoder_num_heads,
        norm_layer=torch.nn.LayerNorm,
        num_frames=params.n_steps,
        drop_path_rate=params.drop_path_rate,
        ssl=params.ssl,
        dropout_p=params.drop_path_rate,   # same as before
    )
    model.default_cfg = _cfg()
    return model
