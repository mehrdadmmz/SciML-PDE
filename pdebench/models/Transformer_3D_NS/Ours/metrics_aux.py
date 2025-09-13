from __future__ import annotations
import logging
import math as mt
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def metric_func(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    if_mean: bool = True,
    Lx: float = 1.0,
    Ly: float = 1.0,
    Lz: float = 1.0,
    iLow: int = 4,
    iHigh: int = 12,
    initial_step: int = 1,
) -> Tuple[torch.Tensor, ...]:

    # identical body
    pred, target = pred.to(device), target.to(device)

    if pred.ndim == target.ndim - 1:               # add dummy T‑dim if needed
        pred = pred.unsqueeze(-2)

    idxs = target.size()
    if len(idxs) == 4:   # 1‑D  (B,X,T,C)
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    elif len(idxs) == 5:  # 2‑D  (B,X,Y,T,C)
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:  # 3‑D  (B,X,Y,Z,T,C)
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)

    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]
    err_mean = torch.sqrt(
        torch.mean((pred.reshape(nb, nc, -1, nt) -
                    target.reshape(nb, nc, -1, nt)) ** 2, dim=2)
    )
    err_RMSE = torch.mean(err_mean, dim=0)         # (C,T)
    nrm = torch.sqrt(torch.mean(target.reshape(nb, nc, -1, nt) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)

    err_CSV = torch.sqrt(
        torch.mean(
            (
                torch.sum(pred.reshape(nb, nc, -1, nt), dim=2)
                - torch.sum(target.reshape(nb, nc, -1, nt), dim=2)
            ) ** 2,
            dim=0,
        )
    )
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    else:  # 3‑D
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz

    err_Max = torch.max(
        torch.max(torch.abs(pred.reshape(nb, nc, -1, nt) -
                            target.reshape(nb, nc, -1, nt)), dim=2)[0],
        dim=0,
    )[0]

    if len(idxs) == 4:  # 1‑D boundaries
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.0), dim=0)
    elif len(idxs) == 5:  # 2‑D boundaries
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) +
                  torch.sum(err_BD_y, dim=-2)) / (2 * nx + 2 * ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    else:  # 3‑D boundaries
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0] - target[:, :, :, :, 0]) ** 2
        err_BD_z += (pred[:, :, :, :, -1] - target[:, :, :, :, -1]) ** 2

        err_BD = (
            torch.sum(err_BD_x.reshape(nb, -1, nt), dim=-2) +
            torch.sum(err_BD_y.reshape(nb, -1, nt), dim=-2) +
            torch.sum(err_BD_z.reshape(nb, -1, nt), dim=-2)
        )

        err_BD = err_BD / (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        err_BD = torch.sqrt(err_BD)

    if len(idxs) == 4:
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = torch.sqrt(torch.mean(
            torch.abs(pred_F - target_F) ** 2, axis=0)) / nx * Lx
    elif len(idxs) == 5:
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt], device=device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i ** 2 + j ** 2))
                if it < err_F.shape[2]:
                    err_F[:, :, it] += _err_F[:, :, i, j]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny) * Lx * Ly
    else:
        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros(
            [nb, nc, min(nx // 2, ny // 2, nz // 2), nt], device=device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                for k in range(nz // 2):
                    it = mt.floor(mt.sqrt(i ** 2 + j ** 2 + k ** 2))
                    if it < err_F.shape[2]:
                        err_F[:, :, it] += _err_F[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / \
            (nx * ny * nz) * Lx * Ly * Lz

    err_F_bands = torch.zeros([nc, 3, nt], device=device)
    err_F_bands[:, 0] = torch.mean(_err_F[:, :iLow], dim=1)
    err_F_bands[:, 1] = torch.mean(_err_F[:, iLow:iHigh], dim=1)
    err_F_bands[:, 2] = torch.mean(_err_F[:, iHigh:], dim=1)

    if if_mean:
        return (
            torch.mean(err_RMSE, dim=[0, -1]),
            torch.mean(err_nRMSE, dim=[0, -1]),
            torch.mean(err_CSV, dim=[0, -1]),
            torch.mean(err_Max, dim=[0, -1]),
            torch.mean(err_BD, dim=[0, -1]),
            torch.mean(err_F_bands, dim=[0, -1]),
        )
    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F_bands


def metrics(
    val_loader,
    model,
    Lx: float,
    Ly: float,
    Lz: float,
    *,
    plot: bool,
    channel_plot: int,
    model_name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    t_min: float,
    t_max: float,
    mode: str = "TransformerAux",
    initial_step: int | None = None,
):
    """Evaluate *model* on *val_loader* that yields primary + auxiliary batches."""
    assert mode == "TransformerAux", "Unsupported mode for metrics_aux.py"

    with torch.no_grad():
        for it, (x_pri, y_pri, x_aux, _, grid) in enumerate(val_loader):
            x_pri = x_pri.to(device, non_blocking=True)       # (B,T,C,X,Y,Z)
            x_aux = x_aux.to(device, non_blocking=True)       # (B,N,T,C,X,Y,Z)
            y_pri = y_pri.to(device, non_blocking=True)       # (B,C,X,Y,Z)

            pred_primary, _ = model(x_pri, grid, x_aux, grid)  # (B,X,Y,Z,1,C)

            pred_r = pred_primary
            yy_r = y_pri.permute(0, 2, 3, 4, 1).unsqueeze(-2)  # (B,X,Y,Z,1,C)

            _eRMSE, _enRMSE, _eCSV, _eMax, _eBD, _eF = metric_func(
                pred_r, yy_r,
                if_mean=True,
                Lx=Lx, Ly=Ly, Lz=Lz,
                initial_step=initial_step or 0,
            )

            if it == 0:
                err_RMSE, err_nRMSE, err_CSV = _eRMSE, _enRMSE, _eCSV
                err_Max,  err_BD,   err_F = _eMax,  _eBD,    _eF
            else:
                err_RMSE += _eRMSE
                err_nRMSE += _enRMSE
                err_CSV += _eCSV
                err_Max += _eMax
                err_BD += _eBD
                err_F += _eF
        it += 1  # number of batches

    err_RMSE = (err_RMSE / it).cpu().numpy()
    err_nRMSE = (err_nRMSE / it).cpu().numpy()
    err_CSV = (err_CSV / it).cpu().numpy()
    err_Max = (err_Max / it).cpu().numpy()
    err_BD = (err_BD / it).cpu().numpy()
    err_F = (err_F / it).cpu().numpy()

    logger.info(f"RMSE: {err_RMSE:.5f}")
    logger.info(f"normalized RMSE: {err_nRMSE:.5f}")
    logger.info(f"RMSE of conserved variables: {err_CSV:.5f}")
    logger.info(f"Maximum value of rms error: {err_Max:.5f}")
    logger.info(f"RMSE at boundaries: {err_BD:.5f}")
    logger.info(f"RMSE in Fourier space: {err_F}")

    if plot:
        dim = yy_r.ndim - 3  # spatial dimensionality (2 or 3)
        with torch.no_grad():
            pred_plot = pred_r[:1].detach().cpu()
            target_plot = yy_r[:1].detach().cpu()

        plt.ioff()
        if dim == 2:  # 2‑D last time slice visualisation
            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(
                pred_plot[0, :, :, -1, channel_plot].T,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                aspect="auto",
            )
            h.set_clim(target_plot[..., channel_plot].min(),
                       target_plot[..., channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Prediction", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            fig.savefig(f"{model_name}_pred.pdf")

            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(
                target_plot[0, :, :, -1, channel_plot].T,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                aspect="auto",
            )
            h.set_clim(target_plot[..., channel_plot].min(),
                       target_plot[..., channel_plot].max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(h, cax=cax)
            cbar.ax.tick_params(labelsize=30)
            ax.set_title("Data", fontsize=30)
            ax.tick_params(axis="x", labelsize=30)
            ax.tick_params(axis="y", labelsize=30)
            ax.set_ylabel("$y$", fontsize=30)
            ax.set_xlabel("$x$", fontsize=30)
            plt.tight_layout()
            fig.savefig(f"{model_name}_data.pdf")

    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


class LpLoss:

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

        diff = torch.norm((xf - yf).view(b, -1), self.p, dim=1)
        norm = eps + torch.norm(yf.view(b, -1), self.p, dim=1)
        if self.reduction == "mean":
            return torch.mean(diff / norm)
        if self.reduction == "sum":
            return torch.sum(diff / norm)
        return diff / norm


class FftMseLoss:

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def __call__(self, x: torch.Tensor, y: torch.Tensor, flow: int | None = None, fhigh: int | None = None):
        b = x.size(0)
        dims = list(range(1, x.dim() - 1))
        xf = torch.fft.fftn(x, dim=dims)
        yf = torch.fft.fftn(y, dim=dims)
        flow = flow or 0
        fhigh = fhigh or min(xf.shape[1:])
        slices = (slice(None),) + (slice(flow, fhigh),) * (x.dim() - 1)
        xf = xf[slices]
        yf = yf[slices]
        diff = (xf - yf).view(b, -1).abs() ** 2
        if self.reduction == "mean":
            return torch.mean(diff)
        if self.reduction == "sum":
            return torch.sum(diff)
        return diff
