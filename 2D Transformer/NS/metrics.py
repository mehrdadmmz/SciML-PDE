# metrics.py ───────────────────────────────────────────────────────────
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
                Lx=1.0, Ly=1.0, Lz=1.0, iLow=4, iHigh=12, initial_step=1):
    """
    computes RMSE, nRMSE, conserved-var error, max error, boundary RMSE,
    and three-band Fourier RMSE.

    Expected tensor shapes:
        [B, X, Y(, Z), T, C]

    but a time length of 1 (T missing) is accepted --> used by Transformer single-step.

    Parameters:
        pred (torch.Tensor): Predicted output
        target (torch.Tensor): Ground truth output
        if_mean (bool): Whether to return mean over batch
        Lx, Ly, Lz (float): Domain lengths
        iLow, iHigh (int): Frequency bands
        initial_step (int): Initial timestep (currently unused)

    Returns:
        tuple of computed metrics
    """
    pred, target = pred.to(device), target.to(device)

    # add dummy time dim if needed (if prediction is 4D)
    if pred.ndim == 4:
        pred = pred.permute(0, 2, 3, 1).unsqueeze(-2)  # [B,H,W,1,C]
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

    idxs = target.size()
    nb, nc, nt = idxs[0], idxs[1], idxs[-1]

    # RMSE
    err_mean = torch.sqrt(
        torch.mean(
            (pred.view([nb, nc, -1, nt]) - target.view([nb, nc, -1, nt])) ** 2, dim=2
        )
    )
    err_RMSE = torch.mean(err_mean, axis=0)
    nrm = torch.sqrt(torch.mean(target.view([nb, nc, -1, nt]) ** 2, dim=2))
    err_nRMSE = torch.mean(err_mean / nrm, dim=0)

    # conserved variables (CSV)
    err_CSV = torch.sqrt(
        torch.mean(
            (
                torch.sum(pred.view([nb, nc, -1, nt]), dim=2)
                - torch.sum(target.view([nb, nc, -1, nt]), dim=2)
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
    elif len(idxs) == 6:
        nx, ny, nz = idxs[2:5]
        err_CSV /= nx * ny * nz

    # Max error
    err_Max = torch.max(
        torch.max(
            torch.abs(pred.view([nb, nc, -1, nt]) -
                      target.view([nb, nc, -1, nt])),
            dim=2,
        )[0],
        dim=0,
    )[0]

    # Boundary error --> RMSE at domain boundaries
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

    # Fourier space RMSE
    if len(idxs) == 4:  # 1D
        nx = idxs[2]
        pred_F = torch.fft.rfft(pred, dim=2)
        target_F = torch.fft.rfft(target, dim=2)
        _err_F = (
            torch.sqrt(torch.mean(
                torch.abs(pred_F - target_F) ** 2, axis=0)) / nx * Lx
        )
    elif len(idxs) == 5:  # 2D
        pred_F = torch.fft.fftn(pred, dim=[2, 3])
        target_F = torch.fft.fftn(target, dim=[2, 3])
        nx, ny = idxs[2:4]
        _err_F = torch.abs(pred_F - target_F) ** 2
        err_F = torch.zeros([nb, nc, min(nx // 2, ny // 2), nt]).to(device)
        for i in range(nx // 2):
            for j in range(ny // 2):
                it = mt.floor(mt.sqrt(i**2 + j**2))
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
                    it = mt.floor(mt.sqrt(i**2 + j**2 + k**2))
                    if it > min(nx // 2, ny // 2, nz // 2) - 1:
                        continue
                    err_F[:, :, it] += _err_F[:, :, i, j, k]
        _err_F = torch.sqrt(torch.mean(err_F, axis=0)) / \
            (nx * ny * nz) * Lx * Ly * Lz

    # Group Fourier errors into bands
    err_F = torch.zeros([nc, 3, nt]).to(device)
    err_F[:, 0] += torch.mean(_err_F[:, :iLow], dim=1)     # low freq
    err_F[:, 1] += torch.mean(_err_F[:, iLow:iHigh], dim=1)  # mid freq
    err_F[:, 2] += torch.mean(_err_F[:, iHigh:], dim=1)    # high freq

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


# metrics function that handles different model types
# (Transformer, FNO, Unet) --> default is Transformer
def metrics(
    val_loader, model,
    Lx, Ly, Lz,
    plot, channel_plot, model_name,
    x_min, x_max, y_min, y_max, t_min, t_max,
    mode="Transformer",                 # default to "Transformer"
    initial_step=None,
):
    """
    Runs over `val_loader` and aggregates metric_func outputs.
    """

    if mode == "Transformer":
        with torch.no_grad():
            for itot, (x_in, y_tgt) in enumerate(val_loader):
                # shapes: [B,T,C,H,W] and [B,C,H,W]
                x_in = x_in.permute(1, 0, 2, 3, 4).to(
                    device, non_blocking=True)  # T,B,C,H,W
                y_tgt = y_tgt.to(device, non_blocking=True)
                # (B,C,H,W)
                pred = model(x_in)

                # reshape so time dim exists
                pred_r = pred.permute(0, 2, 3, 1).unsqueeze(-2)    # B,H,W,1,C
                y_tgt_r = y_tgt.permute(0, 2, 3, 1).unsqueeze(-2)

                (_err_RMSE, _err_nRMSE, _err_CSV,
                 _err_Max, _err_BD, _err_F) = metric_func(
                     pred_r, y_tgt_r, if_mean=True,
                     Lx=Lx, Ly=Ly, Lz=Lz, initial_step=0
                )

                if itot == 0:
                    err_RMSE = _err_RMSE
                    err_nRMSE = _err_nRMSE
                    err_CSV = _err_CSV
                    err_Max = _err_Max
                    err_BD = _err_BD
                    err_F = _err_F
                else:
                    err_RMSE += _err_RMSE
                    err_nRMSE += _err_nRMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F
        itot += 1

    elif mode == "FNO":
        with torch.no_grad():
            itot = 0
            for itot, (xx, yy, grid) in enumerate(val_loader):
                xx = xx.to(device)  # noqa: PLW2901
                yy = yy.to(device)  # noqa: PLW2901
                grid = grid.to(device)  # noqa: PLW2901

                # pred = yy[..., :initial_step, :]
                # inp_shape = list(xx.shape)
                # inp_shape = inp_shape[:-2]
                # inp_shape.append(-1)

                # for _t in range(initial_step, yy.shape[-2]):
                #     # inp = xx.reshape(inp_shape)
                pred = model(xx, grid)
                #     pred = torch.cat((pred, im), -2)
                #     # xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # noqa: PLW2901
                #     xx = torch.cat((xx[..., 1:, :], yy[..., _t+1 : _t+2, :]), dim=-2)  # noqa: PLW2901

                (
                    _err_RMSE,
                    _err_nRMSE,
                    _err_CSV,
                    _err_Max,
                    _err_BD,
                    _err_F,
                ) = metric_func(
                    pred,
                    yy,
                    if_mean=True,
                    Lx=Lx,
                    Ly=Ly,
                    Lz=Lz,
                    initial_step=initial_step,
                )
                if itot == 0:
                    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = (
                        _err_RMSE,
                        _err_nRMSE,
                        _err_CSV,
                        _err_Max,
                        _err_BD,
                        _err_F,
                    )
                    pred_plot = pred[:1]
                    target_plot = yy[:1]
                    val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                else:
                    err_RMSE += _err_RMSE
                    err_nRMSE += _err_nRMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F

                    mean_dim = list(range(len(yy.shape) - 2))
                    mean_dim.append(-1)
                    mean_dim = tuple(mean_dim)
                    val_l2_time += torch.sqrt(
                        torch.mean((pred - yy) ** 2, dim=mean_dim)
                    )

    elif mode == "Unet":
        with torch.no_grad():
            for itot, (xx, yy) in enumerate(val_loader):
                xx = xx.to(device)  # noqa: PLW2901
                yy = yy.to(device)  # noqa: PLW2901

                pred = yy[..., :initial_step, :]
                inp_shape = list(xx.shape)
                inp_shape = inp_shape[:-2]
                inp_shape.append(-1)

                for _t in range(initial_step, yy.shape[-2]):
                    inp = xx.reshape(inp_shape)
                    temp_shape = [0, -1]
                    temp_shape.extend(list(range(1, len(inp.shape) - 1)))
                    inp = inp.permute(temp_shape)

                    temp_shape = [0]
                    temp_shape.extend(list(range(2, len(inp.shape))))
                    temp_shape.append(1)
                    im = model(inp).permute(temp_shape).unsqueeze(-2)
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., 1:, :], im), dim=-2)  # noqa: PLW2901

                (
                    _err_RMSE,
                    _err_nRMSE,
                    _err_CSV,
                    _err_Max,
                    _err_BD,
                    _err_F,
                ) = metric_func(
                    pred,
                    yy,
                    if_mean=True,
                    Lx=Lx,
                    Ly=Ly,
                    Lz=Lz,
                    initial_step=initial_step,
                )

                if itot == 0:
                    err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F = (
                        _err_RMSE,
                        _err_nRMSE,
                        _err_CSV,
                        _err_Max,
                        _err_BD,
                        _err_F,
                    )
                    pred_plot = pred[:1]
                    target_plot = yy[:1]
                    val_l2_time = torch.zeros(yy.shape[-2]).to(device)
                else:
                    err_RMSE += _err_RMSE
                    err_nRMSE += _err_nRMSE
                    err_CSV += _err_CSV
                    err_Max += _err_Max
                    err_BD += _err_BD
                    err_F += _err_F

                    mean_dim = list(range(len(yy.shape) - 2))
                    mean_dim.append(-1)
                    mean_dim = tuple(mean_dim)
                    val_l2_time += torch.sqrt(
                        torch.mean((pred - yy) ** 2, dim=mean_dim)
                    )

    else:
        raise ValueError(f"Unknown metrics mode: {mode}")

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

    return err_RMSE, err_nRMSE, err_CSV, err_Max, err_BD, err_F


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
        super().__init__()
        # Dimension and Lp-norm type are positive
        assert p > 0
        self.p = p
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
