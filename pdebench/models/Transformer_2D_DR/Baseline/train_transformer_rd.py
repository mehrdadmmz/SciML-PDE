# train_transformer_rd.py

from __future__ import annotations
import gc
import random
import pickle
import time
import math
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from utils import ReactionDiffusionTransformerDataset
from transformer_rd import ViT2d
from metrics import FftLpLoss, LpLoss, metrics


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps: int,
                 total_steps: int, min_lr: float = 0.0):
        self.opt = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_n = 0

    def step(self):
        self.step_n += 1
        if self.step_n < self.warmup_steps:
            scale = self.step_n / self.warmup_steps
        else:
            progress = (self.step_n - self.warmup_steps) / \
                max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for g, base in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = self.min_lr + (base - self.min_lr) * scale


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def nrmse(pred: torch.Tensor, tgt: torch.Tensor):
    """True pixel-wise nRMSE (same definition as metrics.py)."""
    spatial = (1, 2, 3)                       # (C,H,W)
    mse = (pred - tgt).pow(2).mean(spatial, keepdim=True)
    denom = tgt.pow(2).mean(spatial, keepdim=True) + 1e-7
    return torch.sqrt(mse) / torch.sqrt(denom)   # ← added sqrt


def run_training(cfg: DictConfig,
                 fourier_alpha: float = 0.1):    # weight of FFT-L2 loss
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    train_ds = ReactionDiffusionTransformerDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        train_subsample=cfg.train_subsample,
        if_test=False)
    val_ds = ReactionDiffusionTransformerDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        if_test=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True)

    print(
        f"train batches: {len(train_loader)} — val batches: {len(val_loader)}")

    tubelet = cfg.tubelet_size
    decoder_nc = tubelet * cfg.in_chans * cfg.patch_size ** 2

    vit_core = ViT2d(
        num_channels=cfg.in_chans,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        tubelet_size=tubelet,
        initial_step=cfg.initial_step,
        encoder_embed_dim=cfg.encoder_embed_dim,
        encoder_depth=cfg.encoder_depth,
        encoder_num_heads=cfg.encoder_num_heads,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_depth=cfg.decoder_depth,
        decoder_num_heads=cfg.decoder_num_heads,
        drop_path_rate=cfg.drop_path_rate,
        ssl=cfg.ssl,
        **{"decoder_num_classes": decoder_nc}
    ).to(dev)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        vit_core = nn.DataParallel(vit_core)

    suffix = "_".join(str(x) for x in cfg.train_subsample)
    model_path = f"{cfg.model_name}_ds{suffix}.pt"

    optim = torch.optim.AdamW(
        vit_core.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay)

    total_steps = cfg.epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_steps_pct * total_steps)
    scheduler = WarmupCosine(optim, warmup_steps, total_steps)

    lp_loss = LpLoss(p=2, reduction="mean")
    fft_loss = FftLpLoss(p=2, reduction="mean")

    wandb.init(project="2D_RD_transformer", config=OmegaConf.to_container(cfg))
    start_time = time.time()
    wandb.define_metric("sim_hours")
    for k in ["train_primary_loss", "val_primary_loss", "learning rate"]:
        wandb.define_metric(k, step_metric="sim_hours")

    best_val, start_ep = float("inf"), 0
    if cfg.continue_training and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=dev)
        vit_core.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        for s in optim.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(dev)
        start_ep, best_val = ckpt["epoch"], ckpt["loss"]

    # evaluation-only path
    if not cfg.if_training:
        vit_core.load_state_dict(torch.load(
            model_path, map_location=dev)["model_state_dict"])
        vit_core.eval()

        err_tuple = metrics(
            val_loader, vit_core,
            Lx=1.0, Ly=1.0, Lz=1.0,
            plot=cfg.plot, channel_plot=cfg.channel_plot,
            model_name=cfg.model_name,
            x_min=cfg.x_min, x_max=cfg.x_max,
            y_min=cfg.y_min, y_max=cfg.y_max,
            t_min=cfg.t_min, t_max=cfg.t_max,
            mode="Transformer",
            initial_step=cfg.initial_step,
        )

        # print & wandb log (same as baseline)
        rmse, nrmse_val, csv, max_, bd, freq_rmse = err_tuple
        print(f"Validation  RMSE {rmse:.5e}  nRMSE {nrmse_val:.5e}  "
              f"CSV {csv:.5e}  Max {max_:.5e}  BD {bd:.5e}")

        wandb.log({
            "val_primary_RMSE":     rmse,
            "val_primary_nRMSE":    nrmse_val,
            "val_primary_CSV":      csv,
            "val_primary_Max":      max_,
            "val_primary_BD":       bd,
            "val_primary_FreqRMSE": freq_rmse,
        })

        # store full tuple for later inspection
        with open(f"{cfg.model_name}_ds{suffix}_eval_metrics.pkl", "wb") as f:
            pickle.dump(err_tuple, f)
        return

    # training loop
    accum_iter = 2
    scaler = torch.cuda.amp.GradScaler()

    for ep in tqdm(range(start_ep, cfg.epochs), desc="Epoch"):
        vit_core.train()
        tr_loss, nbatches = 0.0, 0
        optim.zero_grad(set_to_none=True)

        for bidx, (x_in, y_tgt, _) in enumerate(train_loader):
            x_in = x_in.to(dev, non_blocking=True)
            y_tgt = y_tgt.to(dev, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                B, T, C, H, W = x_in.shape
                x_flat = x_in.permute(
                    0, 3, 4, 1, 2).contiguous().view(B, H, W, T * C)
                y_pred = vit_core(x_flat)
                loss_pix = nrmse(y_pred, y_tgt).mean()
                loss_fft = fft_loss(y_pred, y_tgt)
                loss = (loss_pix + fourier_alpha * loss_fft) / accum_iter

            scaler.scale(loss).backward()
            nbatches += 1

            if (bidx + 1) % accum_iter == 0 or (bidx + 1) == len(train_loader):
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(vit_core.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            tr_loss += loss.item() * accum_iter
        tr_loss /= nbatches

        # validation
        vit_core.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            for x_in, y_tgt, _ in val_loader:
                x_in = x_in.to(dev, non_blocking=True)
                y_tgt = y_tgt.to(dev, non_blocking=True)
                B, T, C, H, W = x_in.shape
                x_flat = x_in.permute(
                    0, 3, 4, 1, 2).contiguous().view(B, H, W, T * C)
                pred = vit_core(x_flat)
                val_loss += nrmse(pred, y_tgt).mean().item()
        val_loss /= len(val_loader)

        # checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": ep,
                        "model_state_dict": vit_core.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "loss": best_val},
                       model_path)

        # logging
        sim_hours = (time.time() - start_time) / 3600
        wandb.log({"sim_hours": sim_hours,
                   "Epoch": ep,
                   "train_primary_loss": tr_loss,
                   "val_primary_loss":   val_loss,
                   "learning rate": optim.param_groups[0]["lr"]})

        print(f"Epoch {ep:03d} — train {tr_loss:.4e} — val {val_loss:.4e}")
        gc.collect()
        torch.cuda.empty_cache()


# Hydra entry-point
@hydra.main(version_base=None, config_path=".", config_name="config_transformer_rd")
def _main(cfg: DictConfig):
    torch.multiprocessing.set_start_method("spawn", force=True)
    run_training(cfg.args if "args" in cfg else cfg)


if __name__ == "__main__":
    _main()
