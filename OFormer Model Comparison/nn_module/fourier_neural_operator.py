# nn_module/fourier_neural_operator.py
# ------------------------------------------------------------
#  Minimal, self-contained 2-D FNO layer needed by OFormer
#  (originally from Li et al. 2021 – MIT licence)
# ------------------------------------------------------------
import torch
import torch.fft
import torch.nn as nn
import math


def compl_mul2d(a: torch.Tensor, b: torch.Tensor):
    # a, b : (..., 2) last dim → real & imag
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],   # real
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],   # imag
    ], dim=-1)


class SpectralConv2d_fast(nn.Module):
    """
    2-D Fourier layer.  Does FFT -→ linear transform in frequency
    domain -→ inverse FFT.  The number of Fourier modes kept in each
    dimension is configurable (defaults: keep all).
    """

    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 modes_x: int = None,
                 modes_y: int = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x    # if None ⇒ keep full spectrum
        self.modes_y = modes_y

       # initialise complex weights  [in_c, out_c, mx, my, 2]
        self.scale = 1 / (in_channels * out_channels)

        def _init(m):
            std = self.scale / \
                math.sqrt(
                    modes_x * modes_y) if (modes_x and modes_y) else self.scale
            return torch.randn(in_channels, out_channels, m, m, 2) * std

        if modes_x is None or modes_y is None:
            # dynamic – allocate later when we know spatial size
            self.weight = None
        else:
            self.weight = nn.Parameter(_init(modes_x))

    # helper: allocate weight if full size is unknown at build time
    def _maybe_init_weight(self, H, W, device):
        if self.weight is not None:
            return
        mx = H // 2 + 1   # rfft size
        my = W // 2 + 1
        self.modes_x, self.modes_y = mx, my
        w = torch.randn(self.in_channels, self.out_channels, mx, my, 2,
                        device=device) * self.scale / math.sqrt(mx * my)
        self.weight = nn.Parameter(w)

    def forward(self, x):                        # x: (B,C,H,W)
        B, C, H, W = x.shape
        self._maybe_init_weight(H, W, x.device)

        # 1) FFT
        # (B, C, H, W/2+1) complex
        x_ft = torch.fft.rfft2(x, norm="forward")
        # convert to real-imag tensor so we can multiply
        x_ft = torch.stack([x_ft.real, x_ft.imag], dim=-1)  # (B,C,H,K,2)

        # 2) Multiply by learned weights (complex multiplication)
        out_ft = compl_mul2d(x_ft[:, :, :self.modes_x, :self.modes_y],
                             self.weight)                 # (B,out_c,…)

        # pad back to full size in frequency domain
        f_pad = torch.zeros(B, self.out_channels, H, W // 2 + 1, 2,
                            device=x.device, dtype=x.dtype)
        f_pad[:, :, :self.modes_x, :self.modes_y] = out_ft

        # 3) iFFT
        out_ft = torch.complex(f_pad[..., 0], f_pad[..., 1])
        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="forward")
        return x_out
