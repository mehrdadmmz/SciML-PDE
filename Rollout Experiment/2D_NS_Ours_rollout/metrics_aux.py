from __future__ import annotations

import logging
import math as mt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn

# device setup ---------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# metric_func  – unchanged numerical core (now accepts 4‑D *or* multi‑frame)
# ─────────────────────────────────────────────────────────────────────────────


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
):
    """Compute standard NS diagnostics on *any* tensor layout.

    The function now seamlessly handles shapes

    * **Single frame**: ``[B, C, H, W]`` – a time‑dim is inserted.
    * **2‑D movie** : ``[B, H, W, T, C]``
    * **3‑D movie** : ``[B, H, W, D, T, C]``

    Everything after the first four lines is byte‑for‑byte identical to the
    original implementation – only the wrapper that adds ``T=1`` when the net
    outputs a single frame was inserted.
    """

    pred, target = pred.to(device), target.to(device)

    # ── add dummy time dim if the network returns a single frame ────────────
    if pred.ndim == 4:  # (B,C,H,W)
        pred = pred.permute(0, 2, 3, 1).unsqueeze(-2)   # B,H,W,1,C
        target = target.permute(0, 2, 3, 1).unsqueeze(-2)

    # ---------------------------------------------------------------------
    #  Below is the *unaltered* metric body from the original file
    # ---------------------------------------------------------------------
    idxs = target.size()
    if len(idxs) == 4:  # 1D
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    if len(idxs) == 5:  # 2D
        pred = pred.permute(0, 4, 1, 2, 3)  # batch, nc, nx, ny, nt
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:  # 3D
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]

    # RMSE ---------------------------------------------------------------
    err_mean = torch.sqrt(
        torch.mean(
            (pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2,
            dim=2,
        )
    )
    err_RMSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)

    err_CSV = torch.sqrt(
        torch.mean(
            (
                torch.sum(pred.view([nb, nc, -1, nt]), dim=2)
                - torch.sum(target.view([nb, nc, -1, nt]), dim=2)
            )
            ** 2,
            dim=0,
        )
    )
    if len(idxs) == 4:
        nx = idxs[2]
        err_CSV /= nx
    elif len(idxs) == 5:
        nx, ny = idxs[2:4]
        err_CSV /= nx * ny
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz

    # worst‑case pointwise error ----------------------------------------
    err_Max = torch.max(
        torch.max(
            torch.abs(pred.view([nb, nc, -1, nt]) -
                      target.view([nb, nc, -1, nt])),
            dim=2,
        )[0],
        dim=0,
    )[0]

    # boundary RMSE ------------------------------------------------------
    if len(idxs) == 4:  # 1D
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.0), dim=0)
    elif len(idxs) == 5:  # 2D
        nx, ny = idxs[2:4]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (
            2 * nx + 2 * ny
        )
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)
    elif len(idxs) == 6:  # 3D
        nx, ny, nz = idxs[2:5]
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0] - target[:, :, :, :, 0]) ** 2
        err_BD_z += (pred[:, :, :, :, -1] - target[:, :, :, :, -1]) ** 2
        err_BD = (
            torch.sum(err_BD_x.contiguous().view([nb, -1, nt]), dim=-2)
            + torch.sum(err_BD_y.contiguous().view([nb, -1, nt]), dim=-2)
            + torch.sum(err_BD_z.contiguous().view([nb, -1, nt]), dim=-2)
        )
        err_BD = err_BD / (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        err_BD = torch.sqrt(err_BD)

    # Fourier‑space three‑band RMSE -------------------------------------
    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = (
            torch.sqrt(torch.mean(
                torch.abs(pred_F - target_F) ** 2, axis=0)) / nx * Lx
        )
    if len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i ** 2 + j ** 2))
                if it > min(nx // 2, ny // 2) - 1:
                    continue
                err_F[:, :, it] += _err_F[:, :, i, j]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / (nx * ny) * Lx * Ly
    elif len(idxs) == 6:  # 3D
        pred_F = torch.fft.fftn(pred, dim=[2, 3, 4])
        target_F = torch.fft.fftn(target, dim=[2, 3, 4])
        nx, ny, nz = idxs[2:5]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros(
            [nb, nc, min(nx // 2, ny // 2, nz // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                for k in range(nz // 2):
                    it = mt.floor(mt.sqrt(i ** 2 + j ** 2 + k ** 2))
                    if it > min(nx // 2, ny // 2, nz // 2) - 1:
                        continue
                    err_F[:, :, it] += _err_F[:, :, i, j, k]
        _err_F = (
            torch.sqrt(torch.mean(err_F, axis=0))
            / (nx * ny * nz)
            * Lx
            * Ly
            * Lz
        )

    err_F = torch.zeros([nc, 3, nt]).to(device)
    err_F[:, 0] += torch.mean(_err_F[:, : iLow], dim=1)  # low‑freq
    err_F[:, 1] += torch.mean(_err_F[:, iLow: iHigh], dim=1)  # mid‑freq
    err_F[:, 2] += torch.mean(_err_F[:, iHigh:], dim=1)  # high‑freq

    if if_mean:
        return (
            torch.mean(err_RMSE, dim=[0, -1]),
            torch.mean(err_nRMSE, dim=[0, -1]),
            torch.mean(err_CSV, dim=[0, -1]),
            torch.mean(err_Max, dim=[0, -1]),
            torch.mean(err_BD, dim=[0, -1]),
            torch.mean(err_F, dim=[0, -1]),
        )
    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


# ─────────────────────────────────────────────────────────────────────────────
# metrics() – rollout‑aware evaluation for the TransformerAux model
# ─────────────────────────────────────────────────────────────────────────────

def metrics(
    val_loader,
    model: nn.Module,
    rollout_test: int,
    Lx: float,
    Ly: float,
    Lz: float,
    plot: bool,
    channel_plot: int,
    model_name: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    t_min: float,
    t_max: float,
    *,
    mode: str = "TransformerAux",
    initial_step: int | None = None,
):
    """Evaluate *rollout* performance of the auxiliary Transformer.

    The routine mirrors the FNO version: we unroll ``rollout_test`` steps
    autoregressively, then compare the **last** predicted frame to the ground
    truth and accumulate per‑timestep MSE for diagnostic plots.
    """

    if mode != "TransformerAux":
        raise ValueError(f"Unknown metrics mode: {mode}")

    with torch.no_grad():
        for itot, (x_p, y_p, x_aux, y_aux) in enumerate(val_loader):
            # Shapes -----------------------------------------------------
            # x_p   : [B, T, C, H, W]
            # y_p   : [B, R, C, H, W]
            # x_aux : [B, N_aux, T, C, H, W]
            # y_aux : [B, N_aux, R, C, H, W]

            # move to GPU ----------------------------------------------
            x_p = x_p.to(device, non_blocking=True)
            y_p = y_p.to(device, non_blocking=True)
            x_aux = x_aux.to(device, non_blocking=True)
            y_aux = y_aux.to(device, non_blocking=True)

            # Autoregressive rollout -----------------------------------
            preds: list[torch.Tensor] = []
            for t in range(rollout_test):
                pred_frame, _ = model(x_p, x_aux)  # [B,C,H,W]
                preds.append(pred_frame)

                # slide the primary window
                x_p = torch.cat((x_p[:, 1:], pred_frame.unsqueeze(1)), dim=1)

                # slide each auxiliary window with *ground‑truth* frames
                next_aux = y_aux[:, :, t].unsqueeze(
                    2)  # [B,N,C,H,W] -> add T‑dim
                x_aux = torch.cat((x_aux[:, :, 1:], next_aux), dim=2)

            pred_seq = torch.stack(preds, dim=1)  # [B, R, C, H, W]

            # Last‑step comparison (4‑D tensors → metric_func handles) --
            pred_last = pred_seq[:, -1]
            target_last = y_p[:, -1]

            (
                _err_RMSE,
                _err_nRMSE,
                _err_CSV,
                _err_Max,
                _err_BD,
                _err_F,
            ) = metric_func(
                pred_last,
                target_last,
                if_mean=True,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
                initial_step=0,
            )

            # accumulate ------------------------------------------------
            if itot == 0:
                err_RMSE, err_nRMSE, err_CSV = _err_RMSE, _err_nRMSE, _err_CSV
                err_Max, err_BD, err_F = _err_Max, _err_BD, _err_F

                # time‑series MSE for plotting -------------------------
                mean_dims = (0, 2, 3, 4)  # average over B, C, H, W
                val_l2_time = torch.sqrt(torch.mean(
                    (pred_seq - y_p) ** 2, dim=mean_dims))

                pred_plot = pred_last[:1]
                target_plot = target_last[:1]
            else:
                err_RMSE += _err_RMSE
                err_nRMSE += _err_nRMSE
                err_CSV += _err_CSV
                err_Max += _err_Max
                err_BD += _err_BD
                err_F += _err_F

                mean_dims = (0, 2, 3, 4)
                val_l2_time += torch.sqrt(torch.mean((pred_seq - y_p)
                                          ** 2, dim=mean_dims))

        # loop completed -----------------------------------------------
        itot += 1  # last value from enumerate is *N‑1*

    # Normalise & log -------------------------------------------------------
    err_RMSE = np.array(err_RMSE.cpu()) / itot
    err_nRMSE = np.array(err_nRMSE.cpu()) / itot
    err_CSV = np.array(err_CSV.cpu()) / itot
    err_Max = np.array(err_Max.cpu()) / itot
    err_BD = np.array(err_BD.cpu()) / itot
    err_F = np.array(err_F.cpu()) / itot

    logger.info(f"RMSE: {err_RMSE:.5f}")
    logger.info(f"normalized RMSE: {err_nRMSE:.5f}")
    logger.info(f"RMSE of conserved variables: {err_CSV:.5f}")
    logger.info(f"Maximum value of rms error: {err_Max:.5f}")
    logger.info(f"RMSE at boundaries: {err_BD:.5f}")
    logger.info(f"RMSE in Fourier space: {err_F}")

    # time‑series plot / .npz --------------------------------------------
    val_l2_time = val_l2_time / itot

    if plot:
        dim = len(target_last.shape) - 2  # spatial dim count (2 → 2‑D)
        plt.ioff()

        if dim == 2:  # 2‑D field visualisation
            # predicted -------------------------------------------------
            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(
                pred_plot[..., channel_plot].squeeze().t().detach().cpu(),
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
            plt.savefig(model_name + "_pred.pdf")

            # ground truth --------------------------------------------
            fig, ax = plt.subplots(figsize=(6.5, 6))
            h = ax.imshow(
                target_plot[..., channel_plot].squeeze().t().detach().cpu(),
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
            plt.savefig(model_name + "_data.pdf")

        # save MSE‑vs‑time to .npz -------------------------------------
        np.savez(
            model_name + "_mse_time.npz",
            t=torch.arange(val_l2_time.numel()).cpu(),
            mse=val_l2_time.detach().cpu(),
        )

    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


# ---------------------------------------------------------------------
# Loss-class definitions (LpLoss, FftLpLoss, FftMseLoss)  ─ UNCHANGED
# ---------------------------------------------------------------------
#  … copy your original definitions verbatim …
# LpLoss Function
class LpLoss:
    """
    Lp loss function
    """

    def __init__(self, p=2, reduction="mean"):
        super().__init__()
        # Dimension and Lp-norm type are positive
        assert p > 0
        self.p = p
        self.reduction = reduction

    def __call__(self, x, y, eps=1e-20):
        num_examples = x.size()[0]
        _diff = x.view(num_examples, -1) - y.view(num_examples, -1)
        _diff = torch.norm(_diff, self.p, 1)
        _norm = eps + torch.norm(y.view(num_examples, -1), self.p, 1)
        if self.reduction in ["mean"]:
            return torch.mean(_diff / _norm)
        if self.reduction in ["sum"]:
            return torch.sum(_diff / _norm)
        return _diff / _norm


# FftLoss Function
class FftLpLoss:
    """
    loss function in Fourier space

    June 2022, F.Alesiani
    """

    def __init__(self, p=2, reduction="mean"):
        self.reduction = reduction

    def __call__(self, x, y, flow=None, fhigh=None, eps=1e-20):
        num_examples = x.size()[0]
        others_dims = x.shape[1:]
        dims = list(range(1, len(x.shape)))
        xf = torch.fft.fftn(x, dim=dims)
        yf = torch.fft.fftn(y, dim=dims)
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = np.max(xf.shape[1:])

        if len(others_dims) == 1:
            xf = xf[:, flow:fhigh]
            yf = yf[:, flow:fhigh]
        if len(others_dims) == 2:
            xf = xf[:, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh]
        if len(others_dims) == 3:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh]
        if len(others_dims) == 4:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]

        _diff = xf - yf.reshape(xf.shape)
        _diff = torch.norm(_diff.reshape(num_examples, -1), self.p, 1)
        _norm = eps + torch.norm(yf.reshape(num_examples, -1), self.p, 1)

        if self.reduction in ["mean"]:
            return torch.mean(_diff / _norm)
        if self.reduction in ["sum"]:
            return torch.sum(_diff / _norm)
        return _diff / _norm


# FftLoss Function
class FftMseLoss:
    """
    loss function in Fourier space

    June 2022, F.Alesiani
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        # Dimension and Lp-norm type are positive
        self.reduction = reduction

    def __call__(self, x, y, flow=None, fhigh=None):
        num_examples = x.size()[0]
        others_dims = x.shape[1:-2]
        for d in others_dims:
            assert d > 1, "we expect the dimension to be the same and greater the 1"
        # print(others_dims)
        dims = list(range(1, len(x.shape) - 1))
        xf = torch.fft.fftn(x, dim=dims)
        yf = torch.fft.fftn(y, dim=dims)
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = np.max(xf.shape[1:])

        if len(others_dims) == 1:
            xf = xf[:, flow:fhigh]
            yf = yf[:, flow:fhigh]
        if len(others_dims) == 2:
            xf = xf[:, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh]
        if len(others_dims) == 3:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh]
        if len(others_dims) == 4:
            xf = xf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]
            yf = yf[:, flow:fhigh, flow:fhigh, flow:fhigh, flow:fhigh]
        _diff = xf - yf
        _diff = _diff.reshape(num_examples, -1).abs() ** 2
        if self.reduction in ["mean"]:
            return torch.mean(_diff).abs()
        if self.reduction in ["sum"]:
            return torch.sum(_diff).abs()
        return _diff.abs()


def inverse_metrics(u0, x, pred_u0, y):
    """
    computes all the metrics in the base and fourier space
    u0: estimated initial condition,
    pred_u0: prediction from the estimated initial condition, pred_u0 = model(u0)
    x: true initial condition
    y: true prediction, y = model(x)

    June 2022, F.Alesiani
    """

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
