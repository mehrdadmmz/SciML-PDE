# metrics_aux.py
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


def metric_func(pred, target, if_mean=True,
                Lx=1.0, Ly=1.0, Lz=1.0, initial_step=1):
    """
    Computes RMSE, nRMSE, CSV, Max error, Boundary RMSE.
    Compatible with [B, X, Y(, Z), T, C] or [B, C, H, W].
    """

    pred, target = pred.to(device), target.to(device)

    if pred.ndim == 4:  # (B, C, H, W)
        pred = pred.permute(0, 2, 3, 1).unsqueeze(-2)     # → (B, H, W, 1, C)
        target = target.permute(0, 2, 3, 1).unsqueeze(-2)

    idxs = target.size()
    if len(idxs) == 4:  # 1D
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
    elif len(idxs) == 5:  # 2D
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
    elif len(idxs) == 6:  # 3D
        pred = pred.permute(0, 5, 1, 2, 3, 4)
        target = target.permute(0, 5, 1, 2, 3, 4)
    else:
        raise RuntimeError("Unsupported tensor rank")

    nb, nc, *spatial_dims, nt = pred.shape

    # RMSE
    err_mean = torch.sqrt(
        torch.mean((pred.reshape(nb, nc, -1, nt) -
                    target.reshape(nb, nc, -1, nt)) ** 2, dim=2)
    )
    err_RMSE = torch.mean(err_mean, dim=0)

    # nRMSE
    norm = torch.sqrt(torch.mean(target.reshape(nb, nc, -1, nt) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / norm, dim=0)

    # CSV
    err_CSV = torch.sqrt(
        torch.mean(
            (torch.sum(pred.reshape(nb, nc, -1, nt), dim=2) -
             torch.sum(target.reshape(nb, nc, -1, nt), dim=2)) ** 2,
            dim=0
        )
    )
    err_CSV /= torch.tensor(spatial_dims).prod()

    # Max error
    err_Max = torch.abs(pred.reshape(nb, nc, -1, nt) -
                        target.reshape(nb, nc, -1, nt)).max(dim=2)[0].max(dim=0)[0]

    # Boundary RMSE
    if len(spatial_dims) == 1:
        (nx,) = spatial_dims
        err_BD = (pred[:, :, 0, :] - target[:, :, 0, :]) ** 2
        err_BD += (pred[:, :, -1, :] - target[:, :, -1, :]) ** 2
        err_BD = torch.mean(torch.sqrt(err_BD / 2.0), dim=0)

    elif len(spatial_dims) == 2:
        nx, ny = spatial_dims
        err_BD_x = (pred[:, :, 0, :, :] - target[:, :, 0, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :] - target[:, :, -1, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :] - target[:, :, :, 0, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :] - target[:, :, :, -1, :]) ** 2
        err_BD = (torch.sum(err_BD_x, dim=-2) + torch.sum(err_BD_y, dim=-2)) / (
            2 * nx + 2 * ny)
        err_BD = torch.mean(torch.sqrt(err_BD), dim=0)

    elif len(spatial_dims) == 3:
        nx, ny, nz = spatial_dims
        err_BD_x = (pred[:, :, 0, :, :, :] - target[:, :, 0, :, :, :]) ** 2
        err_BD_x += (pred[:, :, -1, :, :, :] - target[:, :, -1, :, :, :]) ** 2
        err_BD_y = (pred[:, :, :, 0, :, :] - target[:, :, :, 0, :, :]) ** 2
        err_BD_y += (pred[:, :, :, -1, :, :] - target[:, :, :, -1, :, :]) ** 2
        err_BD_z = (pred[:, :, :, :, 0, :] - target[:, :, :, :, 0, :]) ** 2
        err_BD_z += (pred[:, :, :, :, -1, :] - target[:, :, :, :, -1, :]) ** 2
        err_BD = (
            torch.sum(err_BD_x.reshape(nb, -1, nt), dim=-2) +
            torch.sum(err_BD_y.reshape(nb, -1, nt), dim=-2) +
            torch.sum(err_BD_z.reshape(nb, -1, nt), dim=-2)
        )
        err_BD /= (2 * nx * ny + 2 * ny * nz + 2 * nz * nx)
        err_BD = torch.sqrt(err_BD)

    else:
        raise RuntimeError("Unsupported spatial dimensions")

    if if_mean:
        return (
            torch.mean(err_RMSE, dim=[0, -1]),
            torch.mean(err_nRMSE, dim=[0, -1]),
            torch.mean(err_CSV, dim=[0, -1]),
            torch.mean(err_Max, dim=[0, -1]),
            torch.mean(err_BD, dim=[0, -1]),
        )
    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD


def metrics(
    val_loader, model,
    Lx, Ly, Lz,
    plot, channel_plot, model_name,
    x_min, x_max, y_min, y_max, t_min, t_max,
    mode="TransformerAux",              # default for aux model
    initial_step=None,
):
    if mode == "TransformerAux":
        with torch.no_grad():
            for itot, (x_p, y_p, x_aux, y_aux, *_grids) in enumerate(val_loader):
                # Move inputs to device
                x_p = x_p.to(device, non_blocking=True)
                x_aux = x_aux.to(device, non_blocking=True)
                y_p = y_p.to(device, non_blocking=True)

                pred_p, _ = model(x_p, x_aux)

                (_err_RMSE, _err_nRMSE, _err_CSV,
                 _err_Max, _err_BD) = metric_func(
                    pred_p, y_p, if_mean=True,
                    Lx=Lx, Ly=Ly, Lz=Lz, initial_step=0,
                )

                if itot == 0:
                    err_RMSE, err_nRMSE, err_CSV = _err_RMSE, _err_nRMSE, _err_CSV
                    err_Max, err_BD = _err_Max, _err_BD
                else:
                    err_RMSE += _err_RMSE
                    err_nRMSE += _err_nRMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
            itot += 1
    else:
        raise ValueError(f"Unknown metrics mode: {mode}")

    # Final aggregation
    err_RMSE = np.array(err_RMSE.cpu()) / itot
    err_nRMSE = np.array(err_nRMSE.cpu()) / itot
    err_CSV = np.array(err_CSV.cpu()) / itot
    err_Max = np.array(err_Max.cpu()) / itot
    err_BD = np.array(err_BD.cpu()) / itot

    logger.info(f"RMSE: {err_RMSE:.5f}")
    logger.info(f"normalized RMSE: {err_nRMSE:.5f}")
    logger.info(f"RMSE of conserved variables: {err_CSV:.5f}")
    logger.info(f"Maximum value of rms error: {err_Max:.5f}")
    logger.info(f"RMSE at boundaries: {err_BD:.5f}")

    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD


class LpLoss:

    def __init__(self, p=2, reduction="mean"):
        super().__init__()
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


class FftLpLoss:

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


class FftMseLoss:

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
