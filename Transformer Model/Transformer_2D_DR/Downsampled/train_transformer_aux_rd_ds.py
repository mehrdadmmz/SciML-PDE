# train_transformer_aux_rd_ds.py

from __future__ import annotations
import gc
import math
import pickle
import random
import time
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb


from utils_downsample import ReactionDiffusionDownsampleDataset
from transformer_aux_rd import ViT2dAux
from metrics_aux import metrics, LpLoss, FftLpLoss


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


class WarmupCosine:
    def __init__(self, opt, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.opt = opt
        self.warm = warmup_steps
        self.total = total_steps
        self.min_lr = min_lr
        self.base = [g["lr"] for g in opt.param_groups]
        self.step_n = 0

    def step(self):
        self.step_n += 1
        if self.step_n < self.warm:
            scale = self.step_n / self.warm
        else:
            prog = (self.step_n - self.warm) / max(1, self.total - self.warm)
            scale = 0.5 * (1.0 + math.cos(math.pi * prog))
        for g, b in zip(self.opt.param_groups, self.base):
            g["lr"] = self.min_lr + (b - self.min_lr) * scale


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def nrmse(pred: torch.Tensor, tgt: torch.Tensor):
    spatial = (1, 2, 3)
    mse = (pred - tgt).pow(2).mean(spatial, keepdim=True)
    denom = tgt.pow(2).mean(spatial, keepdim=True) + 1e-7
    return torch.sqrt(mse) / torch.sqrt(denom)


def nrmse_aux(pred: torch.Tensor, tgt: torch.Tensor):
    spatial = (1, 2, 3)
    mse = (pred - tgt).pow(2).mean(spatial, keepdim=True)
    denom = tgt.pow(2).mean(spatial, keepdim=True) + 1e-7
    return torch.sqrt(mse) / torch.sqrt(denom)


def run_training(cfg: DictConfig):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    ds_kwargs = dict(
        primary_filename="2D_diff-react_test_all",
        auxiliary_filename="2D_diff-react_test_diff",
        downsample_filename="2D_diff-react_decomp_downsample",
        saved_folder=cfg.base_path,
        initial_step=cfg.initial_step,
        train_subsample=cfg.train_subsample,
        num_aux_samples=cfg.num_aux_samples,
        if_downsample=cfg.if_downsample,
    )

    train_ds = ReactionDiffusionDownsampleDataset(**ds_kwargs, if_test=False)
    val_ds = ReactionDiffusionDownsampleDataset(**ds_kwargs, if_test=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    print(
        f"train batches: {len(train_loader)} — val batches: {len(val_loader)}")

    decoder_nc = cfg.tubelet_size * cfg.in_chans * cfg.patch_size ** 2
    vit_aux = ViT2dAux(
        num_channels=cfg.in_chans,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        tubelet_size=cfg.tubelet_size,
        initial_step=cfg.initial_step,
        encoder_embed_dim=cfg.encoder_embed_dim,
        encoder_depth=cfg.encoder_depth,
        encoder_num_heads=cfg.encoder_num_heads,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_depth=cfg.decoder_depth,
        decoder_num_heads=cfg.decoder_num_heads,
        drop_path_rate=cfg.drop_path_rate,
        decoder_num_classes=decoder_nc,
    ).to(dev)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        vit_aux = nn.DataParallel(vit_aux)

    suffix = "_".join(str(x) for x in cfg.train_subsample)
    model_path = f"{cfg.model_name}_ds{suffix}.pt"

    optim = torch.optim.AdamW(
        vit_aux.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_steps_pct * total_steps)
    scheduler = WarmupCosine(optim, warmup_steps, total_steps)

    swa_start_ep = int(cfg.epochs * 0.9)
    swa_model = AveragedModel(vit_aux)
    swa_scheduler = SWALR(optim, swa_lr=cfg.learning_rate * 0.1)

    fft_loss = FftLpLoss(p=2, reduction="mean")
    wandb.init(project="2D_RD_Downsampled_transformer",
               config=OmegaConf.to_container(cfg))
    start_time = time.time()
    wandb.define_metric("sim_hours")
    for k in ["train_primary_loss", "train_aux_loss",
              "val_primary_loss", "val_aux_loss", "learning rate"]:
        wandb.define_metric(k, step_metric="sim_hours")

    best_val, start_ep = float("inf"), 0
    if cfg.continue_training and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=dev)
        vit_aux.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        for s in optim.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(dev)
        start_ep, best_val = ckpt["epoch"], ckpt["loss"]

    if not cfg.if_training:
        vit_aux.load_state_dict(torch.load(
            model_path, map_location=dev)["model_state_dict"])
        vit_aux.eval()
        err_tuple = metrics(
            val_loader, vit_aux,
            Lx=1.0, Ly=1.0, Lz=1.0,
            plot=cfg.plot, channel_plot=cfg.channel_plot,
            model_name=cfg.model_name,
            x_min=cfg.x_min, x_max=cfg.x_max,
            y_min=cfg.y_min, y_max=cfg.y_max,
            t_min=cfg.t_min, t_max=cfg.t_max,
            mode="ViT2dAux", initial_step=cfg.initial_step)
        rmse, nrmse_val, csv, max_, bd, freq_rmse = err_tuple
        print(f"Validation  RMSE {rmse:.5e}  nRMSE {nrmse_val:.5e}  "
              f"CSV {csv:.5e}  Max {max_:.5e}  BD {bd:.5e}")
        wandb.log({
            "val_primary_RMSE": rmse,
            "val_primary_nRMSE": nrmse_val,
            "val_primary_CSV": csv,
            "val_primary_Max": max_,
            "val_primary_BD": bd,
            "val_primary_FreqRMSE": freq_rmse,
        })
        with open(f"{cfg.model_name}_ds{suffix}_eval_metrics.pkl", "wb") as f:
            pickle.dump(err_tuple, f)
        return

    accum_iter = 2
    scaler = torch.cuda.amp.GradScaler()
    aux_w = cfg.auxiliary_weight

    for ep in tqdm(range(start_ep, cfg.epochs), desc="Epoch"):
        vit_aux.train()
        tr_p = tr_a = nbatches = 0.0
        optim.zero_grad(set_to_none=True)

        for bidx, (x_p, y_p, x_aux, y_aux, _g, _g_aux) in enumerate(train_loader):
            B, N_aux, T, C, H, W = x_aux.shape
            x_p = x_p.to(dev).permute(1, 0, 2, 3, 4)
            y_p = y_p.to(dev)
            x_aux_flat = x_aux.to(dev).reshape(
                B * N_aux, T, C, H, W).permute(1, 0, 2, 3, 4)
            y_aux_flat = y_aux.to(dev).reshape(B * N_aux, C, H, W)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                pred_p, pred_aux = vit_aux(x_p, None, x_aux_flat, None)
                loss_p = nrmse(pred_p, y_p).mean()
                loss_a = nrmse_aux(pred_aux, y_aux_flat).mean()
                loss = (loss_p + aux_w * loss_a) / accum_iter

            scaler.scale(loss).backward()
            nbatches += 1

            if (bidx + 1) % accum_iter == 0 or (bidx + 1) == len(train_loader):
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(vit_aux.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

                # LR schedulers
                if ep >= swa_start_ep:
                    swa_scheduler.step()
                else:
                    scheduler.step()

            tr_p += loss_p.item()
            tr_a += loss_a.item()
        tr_p /= nbatches
        tr_a /= nbatches

        vit_aux.eval()
        val_p = val_a = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for x_p, y_p, x_aux, y_aux, _g, _g_aux in val_loader:
                B, N_aux, T, C, H, W = x_aux.shape
                x_p = x_p.to(dev).permute(1, 0, 2, 3, 4)
                y_p = y_p.to(dev)
                x_aux_flat = x_aux.to(dev).reshape(
                    B * N_aux, T, C, H, W).permute(1, 0, 2, 3, 4)
                y_aux_flat = y_aux.to(dev).reshape(B * N_aux, C, H, W)
                pred_p, pred_aux = vit_aux(x_p, None, x_aux_flat, None)
                val_p += nrmse(pred_p, y_p).mean().item()
                val_a += nrmse_aux(pred_aux, y_aux_flat).mean().item()
        val_p /= len(val_loader)
        val_a /= len(val_loader)

        # SWA: accumulate parameters
        if ep >= swa_start_ep:
            swa_model.update_parameters(vit_aux)

        # checkpoint
        if val_p < best_val:
            best_val = val_p
            torch.save({"epoch": ep,
                        "model_state_dict": vit_aux.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "loss": best_val}, model_path)

        # logging
        sim_hours = (time.time() - start_time) / 3600
        wandb.log({"sim_hours": sim_hours, "Epoch": ep,
                   "train_primary_loss": tr_p, "train_aux_loss": tr_a,
                   "val_primary_loss": val_p, "val_aux_loss": val_a,
                   "learning rate": optim.param_groups[0]["lr"]})
        print(f"Epoch {ep:03d} — train P {tr_p:.4e} A {tr_a:.4e} "
              f"— val P {val_p:.4e} A {val_a:.4e}")
        gc.collect()
        torch.cuda.empty_cache()

    print("\nUpdating batch-norm statistics for SWA model …")
    update_bn(train_loader, swa_model, device=dev)

    swa_model.eval()
    val_p = val_a = 0.0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for x_p, y_p, x_aux, y_aux, _g, _g_aux in val_loader:
            B, N_aux, T, C, H, W = x_aux.shape
            x_p = x_p.to(dev).permute(1, 0, 2, 3, 4)
            y_p = y_p.to(dev)
            x_aux_flat = x_aux.to(dev).reshape(
                B * N_aux, T, C, H, W).permute(1, 0, 2, 3, 4)
            y_aux_flat = y_aux.to(dev).reshape(B * N_aux, C, H, W)
            pred_p, pred_aux = swa_model(x_p, None, x_aux_flat, None)
            val_p += nrmse(pred_p, y_p).mean().item()
            val_a += nrmse_aux(pred_aux, y_aux_flat).mean().item()
    val_p /= len(val_loader)
    val_a /= len(val_loader)

    print(f"SWA — val P {val_p:.4e}  A {val_a:.4e}")
    wandb.log({"SWA_val_primary_loss": val_p, "SWA_val_aux_loss": val_a})
    torch.save(swa_model.state_dict(),
               f"{cfg.model_name}_ds{suffix}_swa.pt")


@hydra.main(version_base=None, config_path=".", config_name="config_transformer_aux_rd_ds")
def _main(cfg: DictConfig):
    torch.multiprocessing.set_start_method("spawn", force=True)
    run_training(cfg.args if "args" in cfg else cfg)


if __name__ == "__main__":
    _main()
