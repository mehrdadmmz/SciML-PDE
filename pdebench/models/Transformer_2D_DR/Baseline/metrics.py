# metrics.py

from __future__ import annotations
import logging
import math as mt
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


#  metric_func


def metric_func(
    pred, target, if_mean=True,
    Lx=1.0, Ly=1.0, Lz=1.0,
    iLow=4, iHigh=12, initial_step=1,
):

    # generic reshaping
    pred, target = pred.to(device), target.to(device)
    if pred.ndim == 4:                       # 2-D single-step from Transformer
        pred = pred.permute(0, 2, 3, 1).unsqueeze(-2)   # B,H,W,1,C
        target = target.permute(0, 2, 3, 1).unsqueeze(-2)

    idxs = target.size()
    if len(idxs) == 4:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    elif len(idxs) == 5:
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)

    nb, nc, nt = pred.size(0), pred.size(1), pred.size(-1)

    # RMSE / nRMSE
    err_mean = torch.sqrt(torch.mean(
        (pred.reshape(nb, nc, -1, nt) - target.reshape(nb, nc, -1, nt))**2, dim=2))
    err_RMSE = torch.mean(err_mean,  axis=0)
    nrm = torch.sqrt(torch.mean(target.reshape(nb, nc, -1, nt)**2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)

    # conserved variables (CSV)
    err_CSV = torch.sqrt(torch.mean(
        (torch.sum(pred.reshape(nb, nc, -1, nt),   dim=2) -
         torch.sum(target.reshape(nb, nc, -1, nt), dim=2))**2, dim=0))

    if len(idxs) == 4:
        err_CSV /= idxs[2]
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx*ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx*ny*nz

    # point-wise max error
    err_Max = torch.max(torch.max(
        torch.abs(pred.reshape(nb, nc, -1, nt) -
                  target.reshape(nb, nc, -1, nt)), dim=2)[0], dim=0)[0]

    # boundary RMSE (handles 1/2/3-D)
    if len(idxs) == 4:          # 1-D
        err_BD = (pred[:, :, 0, :]-target[:, :, 0, :])**2 + \
                 (pred[:, :, -1, :]-target[:, :, -1, :])**2
        err_BD = torch.mean(torch.sqrt(err_BD/2.0), dim=0)
    elif len(idxs) == 5:        # 2-D
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :]-target[:, :, 0, :, :])**2 + \
                   (pred[:, :, -1, :, :]-target[:, :, -1, :, :])**2
        err_BD_y = (pred[:, :, :, 0, :]-target[:, :, :, 0, :])**2 + \
                   (pred[:, :, :, -1, :]-target[:, :, :, -1, :])**2
        err_BD = (torch.sum(err_BD_x, dim=-2) +
                  torch.sum(err_BD_y, dim=-2)) / (2*nx+2*ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    else:                       # 3-D
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :, :]-target[:, :, 0, :, :, :])**2 + \
                   (pred[:, :, -1, :, :, :]-target[:, :, -1, :, :, :])**2
        err_BD_y = (pred[:, :, :, 0, :, :]-target[:, :, :, 0, :, :])**2 + \
                   (pred[:, :, :, -1, :, :]-target[:, :, :, -1, :, :])**2
        err_BD_z = (pred[:, :, :, :, 0, :]-target[:, :, :, :, 0, :])**2 + \
                   (pred[:, :, :, :, -1, :]-target[:, :, :, :, -1, :])**2
        err_BD = (torch.sum(err_BD_x.reshape(nb, -1, nt), dim=-2) +
                  torch.sum(err_BD_y.reshape(nb, -1, nt), dim=-2) +
                  torch.sum(err_BD_z.reshape(nb, -1, nt), dim=-2))
        err_BD = err_BD / (2*nx*ny + 2*ny*nz + 2*nz*nx)
        err_BD = torch.sqrt(err_BD)

    # Fourier RMSE grouped into bands (low/mid/high)
    if len(idxs) == 4:                      # 1-D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred,   dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = torch.sqrt(torch.mean(
            torch.abs(pred_F-target_F)**2, axis=0)) / nx * Lx
    elif len(idxs) == 5:                    # 2-D
        pred_F = torch.fft.fftn(pred,   dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        tmp = torch.abs(pred_F-target_F)**2
        err_F = torch.zeros([nb, nc, min(nx//2, ny//2), nt], device=device)
        for i in range(nx//2):
            for j in range(ny//2):
                k = mt.floor(mt.sqrt(i**2+j**2))
                if k < err_F.size(2):
                    err_F[:, :, k] += tmp[:, :, i, j]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0))/(nx*ny)*Lx*Ly
    else:                                    # 3-D
        pred_F = torch.fft.fftn(pred,   dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        tmp = torch.abs(pred_F-target_F)**2
        err_F = torch.zeros(
            [nb, nc, min(nx//2, ny//2, nz//2), nt], device=device)
        for i in range(nx//2):
            for j in range(ny//2):
                for k in range(nz//2):
                    r = mt.floor(mt.sqrt(i**2+j**2+k**2))
                    if r < err_F.size(2):
                        err_F[:, :, r] += tmp[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0))/(nx*ny*nz)*Lx*Ly*Lz

    err_F = torch.zeros([nc, 3, nt], device=device)
    err_F[:, 0] += torch.mean(_err_F[:, :iLow], dim=1)         # low
    err_F[:, 1] += torch.mean(_err_F[:, iLow:iHigh], dim=1)    # mid
    err_F[:, 2] += torch.mean(_err_F[:, iHigh:], dim=1)        # high

    if if_mean:
        return (
            torch.mean(err_RMSE, dim=[0, -1]),
            torch.mean(err_nRMSE, dim=[0, -1]),
            torch.mean(err_CSV,  dim=[0, -1]),
            torch.mean(err_Max,  dim=[0, -1]),
            torch.mean(err_BD,   dim=[0, -1]),
            torch.mean(err_F,    dim=[0, -1]),
        )
    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


# metrics – aggregates over val_loader for 2D-Transformer


def metrics(
    val_loader, model,
    Lx, Ly, Lz,
    plot, channel_plot, model_name,
    x_min, x_max, y_min, y_max, t_min, t_max,
    mode="Transformer",
    initial_step=None,
):

    if mode == "Transformer":
        with torch.no_grad():
            for itot, (x_in, y_tgt, _grid) in enumerate(val_loader):
                # x_in  : [B, T, C, H, W]  →  (T, B, C, H, W)
                x_in = x_in.to(device, non_blocking=True).permute(
                    1, 0, 2, 3, 4)
                y_tgt = y_tgt.to(device, non_blocking=True)

                pred = model(x_in)                       # (B, C, H, W)

                # reshape to [B, H, W, 1, C] for metric_func
                pred_r = pred.permute(0, 2, 3, 1).unsqueeze(-2)
                y_tgt_r = y_tgt.permute(0, 2, 3, 1).unsqueeze(-2)

                _vals = metric_func(pred_r, y_tgt_r, if_mean=True,
                                    Lx=Lx, Ly=Ly, Lz=Lz, initial_step=0)

                if itot == 0:
                    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = _vals
                else:
                    err_RMSE += _vals[0]
                    err_nRMSE += _vals[1]
                    err_CSV += _vals[2]
                    err_Max += _vals[3]
                    err_BD += _vals[4]
                    err_F += _vals[5]
        itot += 1

    else:
        raise ValueError(f"Unknown metrics mode: {mode}")

    # average over loader
    err_RMSE = (err_RMSE / itot).cpu().numpy()
    err_nRMSE = (err_nRMSE / itot).cpu().numpy()
    err_CSV = (err_CSV / itot).cpu().numpy()
    err_Max = (err_Max / itot).cpu().numpy()
    err_BD = (err_BD / itot).cpu().numpy()
    err_F = (err_F / itot).cpu().numpy()

    logger.info(f"RMSE: {err_RMSE:.5f}")
    logger.info(f"normalized RMSE: {err_nRMSE:.5f}")
    logger.info(f"RMSE of conserved variables: {err_CSV:.5f}")
    logger.info(f"Maximum value of rms error: {err_Max:.5f}")
    logger.info(f"RMSE at boundaries: {err_BD:.5f}")
    logger.info(f"RMSE in Fourier space: {err_F}")

    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


class LpLoss:
    """Lp‑normalised error"""

    def __init__(self, p: int = 2, reduction: str = "mean"):
        assert p > 0
        self.p = p
        self.reduction = reduction

    def __call__(self, x: torch.Tensor, y: torch.Tensor, eps: float = 1.0e-20):
        num = x.size(0)
        diff = x.view(num, -1) - y.view(num, -1)
        diff = torch.norm(diff, self.p, dim=1)
        norm = eps + torch.norm(y.view(num, -1), self.p, dim=1)
        if self.reduction == "mean":
            return torch.mean(diff / norm)
        if self.reduction == "sum":
            return torch.sum(diff / norm)
        return diff / norm


class FftLpLoss:
    """Lp loss in Fourier space"""

    def __init__(self, p: int = 2, reduction: str = "mean"):
        assert p > 0
        self.p = p
        self.reduction = reduction

    def __call__(self, x: torch.Tensor, y: torch.Tensor, flow: int | None = None, fhigh: int | None = None, eps: float = 1.0e-20):
        b = x.size(0)
        dims = list(range(1, x.dim()))
        xf = torch.fft.fftn(x, dim=dims)
        yf = torch.fft.fftn(y, dim=dims)

        flow = flow or 0
        fhigh = fhigh or min(xf.shape[1:])
        slices = (slice(None),) + (slice(flow, fhigh),) * (x.dim() - 1)
        xf = xf[slices]
        yf = yf[slices]

        diff = torch.norm((xf - yf).reshape(b, -1), self.p, dim=1)
        norm = eps + torch.norm(yf.reshape(b, -1), self.p, dim=1)
        if self.reduction == "mean":
            return torch.mean(diff / norm)
        if self.reduction == "sum":
            return torch.sum(diff / norm)
        return diff / norm


class FftMseLoss:
    def __init__(self, reduction="mean"):  self.reduction = reduction

    def __call__(self, x, y, flow=None, fhigh=None):
        n = x.size(0)
        dims = list(range(1, x.ndim-1))
        xf = torch.fft.fftn(x, dim=dims)
        yf = torch.fft.fftn(y, dim=dims)
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = max(x.shape[1:])
        slices = [slice(flow, fhigh)] * (x.ndim-2)
        xf, yf = xf[(slice(None), *slices)], yf[(slice(None), *slices)]
        diff = (xf - yf).reshape(n, -1).abs()**2
        if self.reduction == "mean":
            return torch.mean(diff).abs()
        if self.reduction == "sum":
            return torch.sum(diff).abs()
        return diff.abs()


def inverse_metrics(u0, x, pred_u0, y):

    mseloss_fn = nn.MSELoss(reduction="mean")
    l2loss_fn = LpLoss(p=2, reduction="mean")
    l3loss_fn = LpLoss(p=3, reduction="mean")

    fftmseloss_fn = FftMseLoss(reduction="mean")
    fftl2loss_fn = FftLpLoss(p=2, reduction="mean")
    fftl3loss_fn = FftLpLoss(p=3, reduction="mean")

    # initial condition
    mseloss_u0 = mseloss_fn(u0.view(1, -1), x.view(1, -1)).item()
    l2loss_u0 = l2loss_fn(u0.view(1, -1), x.view(1, -1)).item()
    l3loss_u0 = l3loss_fn(u0.view(1, -1), x.view(1, -1)).item()

    fmid = u0.shape[1] // 4

    fftmseloss_u0 = fftmseloss_fn(u0, x).item()
    fftmseloss_low_u0 = fftmseloss_fn(u0, x, 0, fmid).item()
    fftmseloss_mid_u0 = fftmseloss_fn(u0, x, fmid, 2 * fmid).item()
    fftmseloss_hi_u0 = fftmseloss_fn(u0, x, 2 * fmid).item()

    fftl2loss_u0 = fftl2loss_fn(u0, x).item()
    fftl2loss_low_u0 = fftl2loss_fn(u0, x, 0, fmid).item()
    fftl2loss_mid_u0 = fftl2loss_fn(u0, x, fmid, 2 * fmid).item()
    fftl2loss_hi_u0 = fftl2loss_fn(u0, x, 2 * fmid).item()

    fftl3loss_u0 = fftl3loss_fn(u0, x).item()
    fftl3loss_low_u0 = fftl3loss_fn(u0, x, 0, fmid).item()
    fftl3loss_mid_u0 = fftl3loss_fn(u0, x, fmid, 2 * fmid).item()
    fftl3loss_hi_u0 = fftl3loss_fn(u0, x, 2 * fmid).item()

    # prediction
    mseloss_pred_u0 = mseloss_fn(
        pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
    l2loss_pred_u0 = l2loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()
    l3loss_pred_u0 = l3loss_fn(pred_u0.reshape(1, -1), y.reshape(1, -1)).item()

    fmid = pred_u0.shape[1] // 4
    pred_u0 = pred_u0.squeeze(-1)
    y = y.squeeze(-1)

    fftmseloss_pred_u0 = fftmseloss_fn(pred_u0, y).item()
    fftmseloss_low_pred_u0 = fftmseloss_fn(pred_u0, y, 0, fmid).item()
    fftmseloss_mid_pred_u0 = fftmseloss_fn(pred_u0, y, fmid, 2 * fmid).item()
    fftmseloss_hi_pred_u0 = fftmseloss_fn(pred_u0, y, 2 * fmid).item()

    fftl2loss_pred_u0 = fftl2loss_fn(pred_u0, y).item()
    fftl2loss_low_pred_u0 = fftl2loss_fn(pred_u0, y, 0, fmid).item()
    fftl2loss_mid_pred_u0 = fftl2loss_fn(pred_u0, y, fmid, 2 * fmid).item()
    fftl2loss_hi_pred_u0 = fftl2loss_fn(pred_u0, y, 2 * fmid).item()

    fftl3loss_pred_u0 = fftl3loss_fn(pred_u0, y).item()
    fftl3loss_low_pred_u0 = fftl3loss_fn(pred_u0, y, 0, fmid).item()
    fftl3loss_mid_pred_u0 = fftl3loss_fn(pred_u0, y, fmid, 2 * fmid).item()
    fftl3loss_hi_pred_u0 = fftl3loss_fn(pred_u0, y, 2 * fmid).item()

    return {
        "mseloss_u0": mseloss_u0,
        "l2loss_u0": l2loss_u0,
        "l3loss_u0": l3loss_u0,
        "mseloss_pred_u0": mseloss_pred_u0,
        "l2loss_pred_u0": l2loss_pred_u0,
        "l3loss_pred_u0": l3loss_pred_u0,
        "fftmseloss_u0": fftmseloss_u0,
        "fftmseloss_low_u0": fftmseloss_low_u0,
        "fftmseloss_mid_u0": fftmseloss_mid_u0,
        "fftmseloss_hi_u0": fftmseloss_hi_u0,
        "fftmseloss_pred_u0": fftmseloss_pred_u0,
        "fftmseloss_low_pred_u0": fftmseloss_low_pred_u0,
        "fftmseloss_mid_pred_u0": fftmseloss_mid_pred_u0,
        "fftmseloss_hi_pred_u0": fftmseloss_hi_pred_u0,
        "fftl2loss_u0": fftl2loss_u0,
        "fftl2loss_low_u0": fftl2loss_low_u0,
        "fftl2loss_mid_u0": fftl2loss_mid_u0,
        "fftl2loss_hi_u0": fftl2loss_hi_u0,
        "fftl2loss_pred_u0": fftl2loss_pred_u0,
        "fftl2loss_low_pred_u0": fftl2loss_low_pred_u0,
        "fftl2loss_mid_pred_u0": fftl2loss_mid_pred_u0,
        "fftl2loss_hi_pred_u0": fftl2loss_hi_pred_u0,
        "fftl3loss_u0": fftl3loss_u0,
        "fftl3loss_low_u0": fftl3loss_low_u0,
        "fftl3loss_mid_u0": fftl3loss_mid_u0,
        "fftl3loss_hi_u0": fftl3loss_hi_u0,
        "fftl3loss_pred_u0": fftl3loss_pred_u0,
        "fftl3loss_low_pred_u0": fftl3loss_low_pred_u0,
        "fftl3loss_mid_pred_u0": fftl3loss_mid_pred_u0,
        "fftl3loss_hi_pred_u0": fftl3loss_hi_pred_u0,
    }
