# train_transformer_aux_ns_3d.py
from __future__ import annotations
import gc
import random
import pickle
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from torch.optim import AdamW
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import (
    LinearLR, CosineAnnealingLR, SequentialLR)

from utils_aux import TransformerDatasetAux
from transformer_3d_aux import Transformer3DAux
from metrics_aux import metrics

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def nrmse(pred: torch.Tensor, tgt: torch.Tensor):
    spatial = tuple(range(2, pred.ndim))
    tgt_norm = tgt.pow(2).mean(spatial, keepdim=True) + 1e-7
    return (pred - tgt).pow(2).mean(spatial, keepdim=True) / tgt_norm


def has_inf_or_nan(t: torch.Tensor):
    return torch.isinf(t).any() or torch.isnan(t).any()


def run_training(cfg: DictConfig):

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    print("DEBUG — patch_size from cfg:", cfg.patch_size,
          "input_size:", cfg.input_size)

    # DATA
    train_ds = TransformerDatasetAux(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        aux_saved_folder=getattr(cfg, "aux_base_path", cfg.base_path),
        train_subsample=cfg.train_subsample,
        num_aux_samples=cfg.num_aux_samples,
        if_test=False,
    )
    val_ds = TransformerDatasetAux(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        aux_saved_folder=getattr(cfg, "aux_base_path", cfg.base_path),
        if_test=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # MODEL
    vit_aux = Transformer3DAux(
        img_size=tuple(cfg.input_size),
        patch_size=tuple(cfg.patch_size),
        tubelet_size=cfg.tubelet_size,
        num_frames=cfg.initial_step,
        in_chans=cfg.in_chans,
        encoder_embed_dim=cfg.encoder_embed_dim,
        decoder_embed_dim=cfg.decoder_embed_dim,
        depth=cfg.depth,
        num_heads=cfg.encoder_num_heads,
        drop_path_rate=cfg.drop_path_rate,
        decoder_depth=cfg.decoder_depth,
        decoder_num_heads=cfg.decoder_num_heads,
    ).to(dev)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        vit_aux = torch.nn.DataParallel(vit_aux)

    # OPTIMISER  
    # two parameter groups – encoder vs decoder/heads
    enc_params, head_params = [], []
    for n, p in vit_aux.named_parameters():
        (enc_params if "vit_core.encoder" in n else head_params).append(p)

    optim = AdamW(
        [
            {"params": enc_params,
             "lr": cfg.learning_rate_share},
            {"params": head_params,
             "lr": cfg.learning_rate_heads},
        ],
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

    iters_per_epoch = len(train_loader)
    total_iters = cfg.epochs * iters_per_epoch
    warmup_iters = int(0.05 * total_iters)

    scheduler = SequentialLR(
        optim,
        [
            LinearLR(optim, start_factor=1e-8, end_factor=1.0,
                     total_iters=warmup_iters),
            CosineAnnealingLR(optim,
                              T_max=max(1, total_iters - warmup_iters),
                              eta_min=1e-7),
        ],
        milestones=[warmup_iters],
    )

    # LOGGING
    suffix = "_".join(str(x) for x in cfg.train_subsample)
    model_path = f"{cfg.model_name}_ds{suffix}.pt"

    wandb.init(project="3D_NS_transformer_aux",
               config=OmegaConf.to_container(cfg))
    wandb.define_metric("sim_hours")
    for k in ["train_primary_loss", "train_aux_loss",
              "val_primary_loss", "val_aux_loss",
              "learning rate", "aux_weight"]:
        wandb.define_metric(k, step_metric="sim_hours")

    start_time = time.time()
    best_val, start_ep = float("inf"), 0

    # RESUME
    if cfg.continue_training and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=dev)
        vit_aux.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        for s in optim.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(dev)
        start_ep, best_val = ckpt["epoch"], ckpt["loss"]

    # EVAL‑ONLY
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
            mode="TransformerAux", initial_step=cfg.initial_step,
        )
        rmse, nrmse_val, csv, max_, bd, f = err_tuple
        print(f"Validation  RMSE {rmse:.5e}  nRMSE {nrmse_val:.5e}  "
              f"CSV {csv:.5e}  Max {max_:.5e}  BD {bd:.5e}")
        wandb.log({
            "val_primary_RMSE": rmse,
            "val_primary_nRMSE": nrmse_val,
            "val_primary_CSV": csv,
            "val_primary_Max": max_,
            "val_primary_BD": bd,
            "val_primary_FreqRMSE": f})
        with open(f"{cfg.model_name}_ds{suffix}_eval_metrics.pkl", "wb") as f:
            pickle.dump(err_tuple, f)
        return

    # TRAIN LOOP
    scaler = GradScaler(enabled=True, init_scale=2**8, growth_interval=1000)
    accum_iter = cfg.grad_accum

    for ep in tqdm(range(start_ep, cfg.epochs), desc="Epoch"):
        vit_aux.train()
        tr_pri = tr_aux = 0.0
        nbatches = 0
        optim.zero_grad(set_to_none=True)

        for bidx, (x_pri, y_pri, x_aux, y_aux, grid) in enumerate(train_loader, 1):

            # dynamic auxiliary weight
            aux_w = cfg.auxiliary_weight * \
                max(0.1, 1 - ep / (0.6 * cfg.epochs))

            # GPU transfer
            x_pri = x_pri.to(dev, non_blocking=True)
            y_pri = y_pri.to(dev, non_blocking=True)
            x_aux = x_aux.to(dev, non_blocking=True)
            y_aux = y_aux.to(dev, non_blocking=True)
            grid = grid.to(dev, non_blocking=True)  # grid unused

            with torch.cuda.amp.autocast(dtype=torch.float16):
                pri_pred, aux_pred = vit_aux(x_pri, grid, x_aux, grid)

                pri_pred = pri_pred.squeeze(-2).permute(0, 4, 1, 2, 3)
                aux_pred = aux_pred.squeeze(-2).permute(0, 4, 1, 2, 3)

                y_pri_t = y_pri
                B, N, C_, X_, Y_, Z_ = y_aux.shape
                y_aux_t = y_aux.reshape(B * N, C_, X_, Y_, Z_)

                loss_primary = nrmse(pri_pred, y_pri_t).mean() / accum_iter
                loss_aux = nrmse(aux_pred, y_aux_t).mean() / accum_iter
                # normalise aux loss by fan‑out
                loss = loss_primary + aux_w * \
                    (loss_aux / cfg.num_aux_samples)

            scaler.scale(loss).backward()
            nbatches += 1

            if bidx % accum_iter == 0 or bidx == len(train_loader):
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(vit_aux.parameters(), 1.0)

                if any(has_inf_or_nan(p.grad) for p in vit_aux.parameters() if p.grad is not None):
                    optim.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            tr_pri += loss_primary.item() * accum_iter
            tr_aux += loss_aux.item() * accum_iter

        tr_pri /= nbatches
        tr_aux /= nbatches

        # VALIDATION
        vit_aux.eval()
        val_pri = val_aux = 0.0
        with torch.no_grad():
            for x_pri, y_pri, x_aux, y_aux, grid in val_loader:
                x_pri = x_pri.to(dev, non_blocking=True)
                y_pri = y_pri.to(dev, non_blocking=True)
                x_aux = x_aux.to(dev, non_blocking=True)
                y_aux = y_aux.to(dev, non_blocking=True)
                grid = grid.to(dev,  non_blocking=True)

                pri_pred, aux_pred = vit_aux(x_pri, grid, x_aux, grid)
                pri_pred = pri_pred.squeeze(-2).permute(0, 4, 1, 2, 3)
                aux_pred = aux_pred.squeeze(-2).permute(0, 4, 1, 2, 3)

                y_pri_t = y_pri
                B, N, C_, X_, Y_, Z_ = y_aux.shape
                y_aux_t = y_aux.reshape(B * N, C_, X_, Y_, Z_)

                val_pri += nrmse(pri_pred, y_pri_t).mean().item()
                val_aux += nrmse(aux_pred, y_aux_t).mean().item()

        val_pri /= len(val_loader)
        val_aux /= len(val_loader)

        # CHECKPOINT
        if val_pri < best_val:
            best_val = val_pri
            torch.save({
                "epoch": ep,
                "model_state_dict": vit_aux.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": best_val},
                model_path)

        # LOGGING
        sim_hours = (time.time() - start_time) / 3600
        wandb.log({
            "sim_hours": sim_hours,
            "Epoch": ep,
            "train_primary_loss": tr_pri,
            "train_aux_loss": tr_aux,
            "val_primary_loss": val_pri,
            "val_aux_loss": val_aux,
            "learning rate": scheduler.get_last_lr()[0],
            "aux_weight": aux_w,
        })
        print(f"Epoch {ep:03d} — train_pri {tr_pri:.4e} — train_aux {tr_aux:.4e} "
              f"— val_pri {val_pri:.4e} — val_aux {val_aux:.4e}")

        gc.collect()
        torch.cuda.empty_cache()


@hydra.main(version_base=None,
            config_path=".", config_name="config_transformer_aux_ns_3d")
def _main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    run_training(cfg.args if "args" in cfg else cfg)


if __name__ == "__main__":
    _main()
