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


# project-local imports
from utils import TransformerDataset
from transformer import PretrainVisionTransformer          # non-aux ViT
from metrics import metrics

# reproducibility helper


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# L2-norm relative error (same as before but for [B,C,H,W])


def nrmse(pred: torch.Tensor, tgt: torch.Tensor):
    spatial = (1, 2, 3)              # drop batch axis
    tgt_norm = tgt.pow(2).mean(spatial, keepdim=True) + 1e-7
    return ((pred - tgt).pow(2).mean(spatial, keepdim=True) / tgt_norm)


# main training routine
def run_training(cfg: DictConfig):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # Dataset
    train_ds = TransformerDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        train_subsample=cfg.train_subsample,
        if_test=False,
    )
    val_ds = TransformerDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
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
    print(
        f"train batches: {len(train_loader)} — val batches: {len(val_loader)}")

    # Model
    model = PretrainVisionTransformer(
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

    # multigpu support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    # model path , ex: Transformer_ds0.5_0.25_24.pt
    suffix = "_".join(str(x) for x in cfg.train_subsample)
    model_path = f"{cfg.model_name}_ds{suffix}.pt"

    # Optimiser
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.epochs * len(train_loader)
        )
    else:
        sched = torch.optim.lr_scheduler.StepLR(
            optim, step_size=cfg.scheduler_step, gamma=cfg.scheduler_gamma
        )

    # W&B logging
    wandb.init(project="2D_NS_transformer",
               config=OmegaConf.to_container(cfg))

    start_time = time.time()
    wandb.define_metric("sim_hours")
    wandb.define_metric("train_primary_loss", step_metric="sim_hours")
    wandb.define_metric("val_primary_loss",   step_metric="sim_hours")
    wandb.define_metric("learning rate",      step_metric="sim_hours")

    best_val = float("inf")
    start_ep = 0

    # re-load checkpoint
    if cfg.continue_training and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(dev)
        start_ep = ckpt["epoch"]
        best_val = ckpt["loss"]

    # Evaluation-only mode
    if not cfg.if_training:
        model.load_state_dict(torch.load(
            model_path, map_location=dev)["model_state_dict"])
        model.eval()

        err_tuple = metrics(
            val_loader, model,
            Lx=1.0, Ly=1.0, Lz=1.0,
            plot=cfg.plot, channel_plot=cfg.channel_plot,
            model_name=cfg.model_name,
            x_min=cfg.x_min, x_max=cfg.x_max,
            y_min=cfg.y_min, y_max=cfg.y_max,
            t_min=cfg.t_min, t_max=cfg.t_max,
            mode="Transformer",
            initial_step=cfg.initial_step,
        )

        names = ["RMSE", "nRMSE", "CSV", "Max", "Boundary", "FourierBands"]
        err = {k: v for k, v in zip(names, err_tuple[:5])}
        err["FourierBands"] = err_tuple[5]

        print("Evaluation metrics:", err)

        wandb.log(err)

        # save `err`
        with open(f"{cfg.model_name}_ds{suffix}_eval_metrics.pkl", "wb") as f:
            pickle.dump(err, f)
        return

    # Training loop
    for ep in tqdm(range(start_ep, cfg.epochs), desc="Epoch"):

        # Train
        model.train()
        tr_loss = 0.0

        for x_in, y_tgt in train_loader:
            x_in = x_in.permute(1, 0, 2, 3, 4).to(
                dev, non_blocking=True)  # T,B,C,H,W
            y_tgt = y_tgt.to(dev, non_blocking=True)

            y_pred = model(x_in)  # (B,C,H,W)
            loss = nrmse(y_pred, y_tgt).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max(5.0, 0.1 *
                                                             torch.norm(torch.stack([p.grad.detach().norm() for p in model.parameters()
                                                                                     if p.grad is not None]), 2)))
            optim.step()
            sched.step()

            tr_loss += loss.item()

        tr_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_in, y_tgt in val_loader:
                x_in = x_in.permute(1, 0, 2, 3, 4).to(dev, non_blocking=True)
                y_tgt = y_tgt.to(dev, non_blocking=True)
                y_pred = model(x_in)
                val_loss += nrmse(y_pred, y_tgt).mean().item()
        val_loss /= len(val_loader)

        # Checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "loss": best_val,
                },
                model_path,
            )

        # Logging
        sim_hours = (time.time() - start_time) / 3600
        wandb.log(
            {
                "sim_hours": sim_hours,
                "Epoch": ep,
                "train_primary_loss": tr_loss,
                "val_primary_loss": val_loss,
                "learning rate": optim.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {ep:03d} — train L2 {tr_loss:.4e} — val L2 {val_loss:.4e}")

        # Memory management
        # (this is a hack to avoid OOM errors on some GPUs)
        gc.collect()
        torch.cuda.empty_cache()


# ───────────────────────────────────────────────────────────────────────
# Hydra entry-point (keeps CLI parity with your FNO script)
# ───────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path=".", config_name="config_transformer_ns")
def _main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    run_training(cfg.args if "args" in cfg else cfg)


if __name__ == "__main__":
    _main()
