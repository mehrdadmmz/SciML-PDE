from __future__ import annotations
import gc
import random
import pickle
from pathlib import Path
from typing import Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb
import time

# local modules
from utils_aux import TransformerAuxDataset
from transformer_aux import PretrainVisionTransformerAux
from metrics_aux import metrics

# reproducibility


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# relative-L2 per-pixel


def nrmse(pred: torch.Tensor, tgt: torch.Tensor):
    spatial = (1, 2, 3)                           # C,H,W
    tgt_norm = tgt.pow(2).mean(spatial, keepdim=True) + 1e-7
    return ((pred-tgt).pow(2).mean(spatial, keepdim=True)/tgt_norm)

# ----------------------------------------------------------------------


def build_optimizer(model: nn.Module, lr_share: float, lr_heads: float):
    back_params, head_params = [], []
    for n, p in model.named_parameters():
        if n.startswith("head_primary") or n.startswith("head_auxiliary"):
            head_params.append(p)
        else:
            back_params.append(p)
    return torch.optim.Adam(
        [
            {"params": back_params,
             "lr": lr_share,
             "weight_decay": 1e-4},
            {"params": head_params,
             "lr": lr_heads,
             "weight_decay": 1e-4},
        ]
    )

# ----------------------------------------------------------------------


def run_training(cfg: DictConfig):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # 1. data -----------------------------------------------------------
    train_ds = TransformerAuxDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        train_subsample=cfg.train_subsample,
        num_aux_samples=cfg.num_aux_samples,
        if_test=False,
    )
    val_ds = TransformerAuxDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        num_aux_samples=cfg.num_aux_samples,
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

    # 2. model ----------------------------------------------------------
    model = PretrainVisionTransformerAux(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        tubelet_size=cfg.tubelet_size,
        in_chans=cfg.in_chans,
        encoder_embed_dim=cfg.encoder_embed_dim,
        encoder_num_heads=cfg.encoder_num_heads,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_num_heads=cfg.decoder_num_heads,
        decoder_depth=cfg.decoder_depth,
        num_frames=cfg.initial_step,
        drop_path_rate=cfg.drop_path_rate,
        ssl=cfg.ssl,
    ).to(dev)

    # Multigpu?
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    # model path , ex: TransformerAux_ds0.5_0.25_24.pt
    suffix = "_".join(str(x) for x in cfg.train_subsample)
    model_path = f"{cfg.model_name}_ds{suffix}.pt"

    # 3. optimiser / scheduler -----------------------------------------
    optim = build_optimizer(
        model, cfg.learning_rate_share, cfg.learning_rate_heads)
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.epochs * len(train_loader)
        )
    else:
        sched = torch.optim.lr_scheduler.StepLR(
            optim, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma
        )

    # 4. wandb ----------------------------------------------------------
    wandb.init(project="2D_NS_transformer_aux",
               config=OmegaConf.to_container(cfg))

    start_time = time.time()                       # wall-clock zero-point
    wandb.define_metric("sim_hours")               # X-axis for the new plots
    wandb.define_metric("train_primary", step_metric="sim_hours")
    wandb.define_metric("train_aux",     step_metric="sim_hours")
    wandb.define_metric("val_primary",   step_metric="sim_hours")
    wandb.define_metric("val_aux",       step_metric="sim_hours")
    wandb.define_metric("lr",            step_metric="sim_hours")

    best_val = float("inf")
    start_ep = 0

    # resume?
    if cfg.continue_training and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        for s in optim.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(dev)
        best_val = ckpt["loss"]
        start_ep = ckpt["epoch"]

    # ───────────────────────────── evaluation only ─────────────────────────────
    if not cfg.if_training:
        model.load_state_dict(
            torch.load(model_path, map_location=dev)["model_state_dict"]
        )
        model.eval()

        # 1. run metrics (still returns a 6-tuple)
        errs_tuple = metrics(
            val_loader, model,
            Lx=1.0, Ly=1.0, Lz=1.0,
            plot=cfg.plot, channel_plot=cfg.channel_plot,
            model_name=cfg.model_name,
            x_min=cfg.x_min, x_max=cfg.x_max,
            y_min=cfg.y_min, y_max=cfg.y_max,
            t_min=cfg.t_min, t_max=cfg.t_max,
            mode="TransformerAux",
            initial_step=cfg.initial_step,
        )

        # 2. give the numbers names
        names = ["RMSE", "nRMSE", "CSV", "Max", "Boundary", "FourierBands"]
        errs = {k: v for k, v in zip(names, errs_tuple[:5])}
        errs["FourierBands"] = errs_tuple[5]          # keep 3-band array

        # 3. nice console read-out
        print("Evaluation metrics:", errs)

        # 4. (optional) log to W&B so the run shows the numbers
        wandb.log(errs)

        # 5. save to disk
        with open(f"{cfg.model_name}_ds{suffix}_eval_metrics.pkl", "wb") as f:
            pickle.dump(errs, f)
        return

    # 5. training loop --------------------------------------------------
    for ep in tqdm(range(start_ep, cfg.epochs), desc="epoch"):
        # ---------- train ----------------------------------------------
        model.train()
        tr_p = 0.
        tr_a = 0.
        for x_p, y_p, x_aux, y_aux in train_loader:
            B, N_aux, T, C, H, W = x_aux.shape

            # ------ primary stream -------
            x_p_t = x_p.permute(1, 0, 2, 3, 4).to(
                dev, non_blocking=True)   # T,B,C,H,W
            y_p = y_p.to(dev, non_blocking=True)

            # ------ auxiliary stream -----
            x_aux_ = x_aux.view(
                B*N_aux, T, C, H, W).permute(1, 0, 2, 3, 4).to(dev)
            y_aux_ = y_aux.view(B*N_aux, C, H, W).to(dev)

            pred_p, pred_a = model(x_p_t, x_aux_)

            loss_p = nrmse(pred_p, y_p).mean()
            loss_a = nrmse(pred_a, y_aux_).mean()
            loss = loss_p + cfg.auxiliary_weight * loss_a

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            sched.step()

            tr_p += loss_p.item()
            tr_a += loss_a.item()
        tr_p /= len(train_loader)
        tr_a /= len(train_loader)

        # ---------- validation -----------------------------------------
        model.eval()
        va_p = 0.
        va_a = 0.
        with torch.no_grad():
            for x_p, y_p, x_aux, y_aux in val_loader:
                B, N_aux, T, C, H, W = x_aux.shape
                x_p_t = x_p.permute(1, 0, 2, 3, 4).to(dev)
                x_aux_ = x_aux.view(
                    B*N_aux, T, C, H, W).permute(1, 0, 2, 3, 4).to(dev)
                y_p = y_p.to(dev)
                y_aux_ = y_aux.view(B*N_aux, C, H, W).to(dev)

                pred_p, pred_a = model(x_p_t, x_aux_)
                va_p += nrmse(pred_p, y_p).mean().item()
                va_a += nrmse(pred_a, y_aux_).mean().item()
        va_p /= len(val_loader)
        va_a /= len(val_loader)

        # checkpoint
        if va_p < best_val:
            best_val = va_p
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "loss": best_val,
            }, model_path)

        # wandb logging
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

        print(f"ep {ep:03d} | train P {tr_p:.3e} A {tr_a:.3e} | "
              f"val P {va_p:.3e} A {va_a:.3e}")

        gc.collect()
        torch.cuda.empty_cache()

# ----------------------------------------------------------------------


@hydra.main(version_base=None, config_path=".", config_name="config_transformer_aux_ns")
def _main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    run_training(cfg.args if "args" in cfg else cfg)


if __name__ == "__main__":
    _main()
