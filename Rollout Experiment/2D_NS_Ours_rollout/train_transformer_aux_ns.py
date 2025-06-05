from __future__ import annotations

import gc
import pickle
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
import torch.multiprocessing as mp
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils_aux import TransformerAuxDataset
from transformer_aux import PretrainVisionTransformerAux
from metrics_aux import metrics


# ───────────────────────── helpers ──────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- handle DataParallel checkpoints ---------------------------------


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove 'module.' prefix that DataParallel adds to parameter names."""
    if all(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def nrmse(pred, tgt):
    spatial = (1, 2, 3)                   # C,H,W
    return (pred - tgt).pow(2).mean(spatial, keepdim=True) / (
        tgt.pow(2).mean(spatial, keepdim=True) + 1e-7
    )


def build_optimizer(model: nn.Module, lr_share: float, lr_heads: float):
    back, heads = [], []
    for n, p in model.named_parameters():
        (heads if n.startswith(("head_primary", "head_auxiliary")) else back).append(p)
    return torch.optim.Adam(
        [
            {"params": back,  "lr": lr_share, "weight_decay": 1e-4},
            {"params": heads, "lr": lr_heads, "weight_decay": 1e-4},
        ]
    )


class BatchFirstWrapper(nn.Module):
    """Convert (B,T,…) → (T,B,…) for the ViT; flatten aux batch."""

    def __init__(self, vit): super().__init__(); self.vit = vit

    def forward(self, x_btchw, x_aux_bntchw):
        x_tbchw = x_btchw.permute(1, 0, 2, 3, 4)              # (T,B,C,H,W)
        B, N, T, C, H, W = x_aux_bntchw.shape
        x_aux_tbchw = (
            x_aux_bntchw.view(B * N, T, C, H, W)
            .permute(1, 0, 2, 3, 4)                           # (T,B*N,C,H,W)
        )
        return self.vit(x_tbchw, x_aux_tbchw)


# ─────────────────────── main routine ───────────────────────
def run_training(cfg: DictConfig):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass

    ROLL_TRAIN, ROLL_EVAL = 1, cfg.rollout_test

    # 1. datasets ------------------------------------------------------
    train_ds = TransformerAuxDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.primary_path,
        aux_saved_folder=cfg.aux_path,
        train_subsample=cfg.train_subsample,
        num_aux_samples=cfg.num_aux_samples,
        rollout_test=ROLL_TRAIN,
        if_test=False,
    )
    val_ds = TransformerAuxDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.primary_path,
        aux_saved_folder=cfg.aux_path,
        num_aux_samples=cfg.num_aux_samples,
        rollout_test=ROLL_TRAIN,
        if_test=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
    )
    print(f"train={len(train_loader)}  val={len(val_loader)}")

    # 2. model ---------------------------------------------------------
    vit = PretrainVisionTransformerAux(
        img_size=cfg.img_size, patch_size=cfg.patch_size, tubelet_size=cfg.tubelet_size,
        in_chans=cfg.in_chans, encoder_embed_dim=cfg.encoder_embed_dim,
        encoder_num_heads=cfg.encoder_num_heads, decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_num_heads=cfg.decoder_num_heads, decoder_depth=cfg.decoder_depth,
        num_frames=cfg.initial_step, drop_path_rate=cfg.drop_path_rate, ssl=cfg.ssl,
    ).to(dev)
    model: nn.Module = BatchFirstWrapper(vit)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    suffix = "_".join(str(x) for x in cfg.train_subsample)
    model_path = f"{cfg.model_name}_ds{suffix}.pt"

    # 3. optimiser / scheduler ----------------------------------------
    optim = build_optimizer(
        model, cfg.learning_rate_share, cfg.learning_rate_heads)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.epochs * len(train_loader)
        )
        if cfg.scheduler == "cosine"
        else torch.optim.lr_scheduler.StepLR(optim, cfg.scheduler_step, cfg.scheduler_gamma)
    )
    scaler = GradScaler(enabled=cfg.use_amp)
    grad_accum = cfg.grad_accum

    # 4. wandb ---------------------------------------------------------
    wandb.init(project="2D_NS_transformer_aux",
               config=OmegaConf.to_container(cfg))
    start_time = time.time()
    wandb.define_metric("sim_hours")
    for k in ["train_primary", "train_aux", "val_primary", "val_aux", "lr"]:
        wandb.define_metric(k, step_metric="sim_hours")

    best_val, start_ep = float("inf"), 0
    if cfg.continue_training and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=dev)
        state = strip_module_prefix(ckpt["model_state_dict"])
        model.load_state_dict(state)
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        for s in optim.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(dev)
        best_val, start_ep = ckpt["loss"], ckpt["epoch"]

    # ─────────────── evaluation-only shortcut ───────────────
    if not cfg.if_training:
        eval_ds = TransformerAuxDataset(
            initial_step=cfg.initial_step,
            saved_folder=cfg.primary_path,
            aux_saved_folder=cfg.aux_path,
            num_aux_samples=cfg.num_aux_samples,
            rollout_test=ROLL_EVAL,
            if_test=True,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True, persistent_workers=True
        )
        state = strip_module_prefix(torch.load(
            model_path, map_location=dev)["model_state_dict"])
        model.load_state_dict(state)
        model.eval()

        errs_tuple = metrics(
            eval_loader, model, ROLL_EVAL,
            Lx=1.0, Ly=1.0, Lz=1.0,
            plot=cfg.plot, channel_plot=cfg.channel_plot,
            model_name=cfg.model_name,
            x_min=cfg.x_min, x_max=cfg.x_max,
            y_min=cfg.y_min, y_max=cfg.y_max,
            t_min=cfg.t_min, t_max=cfg.t_max,
            mode="TransformerAux",
            initial_step=cfg.initial_step,
        )
        names = ["RMSE", "nRMSE", "CSV", "Max", "Boundary", "FourierBands"]
        errs = {k: v for k, v in zip(names, errs_tuple[:5])}
        errs["FourierBands"] = errs_tuple[5]
        print("Evaluation metrics:", errs)
        wandb.log(errs)
        with open(f"{cfg.model_name}_ds{suffix}_eval_metrics.pkl", "wb") as f:
            pickle.dump(errs, f)
        return

    # 5. training loop --------------------------------------------------
    for ep in tqdm(range(start_ep, cfg.epochs), desc="epoch"):
        model.train()
        tr_p = tr_a = 0.0
        optim.zero_grad(set_to_none=True)

        for step, (x_p, y_p, x_aux, y_aux) in enumerate(train_loader):
            B, N, _, C, H, W = x_aux.shape
            x_p = x_p.to(dev, non_blocking=True)
            x_aux = x_aux.to(dev, non_blocking=True)
            y_p = y_p.squeeze(1).to(dev, non_blocking=True)     # [B,C,H,W]
            y_aux = y_aux.squeeze(2).to(dev, non_blocking=True)   # [B,N,C,H,W]
            y_aux_flat = y_aux.view(B * N, C, H, W)

            with autocast(enabled=cfg.use_amp):
                pred_p, pred_a = model(x_p, x_aux)
                loss_p = nrmse(pred_p, y_p).mean()
                loss_a = nrmse(pred_a, y_aux_flat).mean()
                loss = loss_p + cfg.auxiliary_weight * loss_a

            (scaler.scale(loss / grad_accum)
             if cfg.use_amp else loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                if cfg.use_amp:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optim.step()
                optim.zero_grad(set_to_none=True)
                sched.step()

            tr_p += loss_p.item()
            tr_a += loss_a.item()

        tr_p /= len(train_loader)
        tr_a /= len(train_loader)

        # ---------- validation ----------------------------------------
        model.eval()
        va_p = va_a = 0.0
        with torch.no_grad():
            for x_p, y_p, x_aux, y_aux in val_loader:
                B, N, _, C, H, W = x_aux.shape
                x_p = x_p.to(dev)
                x_aux = x_aux.to(dev)
                y_p = y_p.squeeze(1).to(dev)
                y_aux = y_aux.squeeze(2).to(dev)
                y_aux_flat = y_aux.view(B * N, C, H, W)
                pred_p, pred_a = model(x_p, x_aux)
                va_p += nrmse(pred_p, y_p).mean().item()
                va_a += nrmse(pred_a, y_aux_flat).mean().item()
        va_p /= len(val_loader)
        va_a /= len(val_loader)

        # best-ckpt -----------------------------------------------------
        if va_p < best_val:
            best_val = va_p
            torch.save(
                {"epoch": ep,
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optim.state_dict(),
                 "loss": best_val},
                model_path,
            )

        # wandb log -----------------------------------------------------
        sim_hours = (time.time() - start_time) / 3600
        wandb.log({
            "sim_hours":     sim_hours,
            "epoch":         ep,
            "train_primary": tr_p,
            "train_aux":     tr_a,
            "val_primary":   va_p,
            "val_aux":       va_a,
            "lr":            optim.param_groups[0]["lr"],
        })

        print(
            f"ep {ep:03d} | train P {tr_p:.3e} A {tr_a:.3e} | val P {va_p:.3e} A {va_a:.3e}"
        )

        gc.collect()
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
@hydra.main(version_base=None, config_path=".", config_name="config_transformer_aux_ns")
def _main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    run_training(cfg.args if "args" in cfg else cfg)


if __name__ == "__main__":
    _main()
