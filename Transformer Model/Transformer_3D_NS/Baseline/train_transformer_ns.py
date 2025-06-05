# train_transformer_ns_3d.py
from __future__ import annotations
import gc
import random
import pickle
import time
from pathlib import Path
from typing import Tuple
import math
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from timm.optim import create_optimizer_v2
from torch.cuda.amp.grad_scaler import GradScaler

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from utils import TransformerDataset                     # 3‑D dataloader
from transformer_3d import PretrainVisionTransformer      # 3‑D ViT  ### 3D CHANGE
from metrics import metrics

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


class BatchFirstWrapper(nn.Module):

    def __init__(self, vit): super().__init__(); self.vit = vit

    def forward(self, x_btcyxz):                # ### 3D CHANGE
        x_tbcyxz = x_btcyxz.permute(1, 0, 2, 3, 4, 5)
        return self.vit(x_tbcyxz)


def has_inf_or_nan(t):
    return torch.isinf(t).any() or torch.isnan(t).any()


def run_training(cfg: DictConfig):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    print("DEBUG — patch_size from cfg:", cfg.patch_size,
          "input_size:", cfg.input_size)

    train_ds = TransformerDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        train_subsample=cfg.train_subsample,
        if_test=False)          # <─ train set

    val_ds = TransformerDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        train_subsample=cfg.train_subsample,
        if_test=True)           # <─ held‑out seeds

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.num_workers,
                                               pin_memory=True,
                                               persistent_workers=True)

    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.num_workers,
                                             pin_memory=True,
                                             persistent_workers=True)
    print(
        f"train batches: {len(train_loader)} — val batches: {len(val_loader)}")

    vit = PretrainVisionTransformer(
        img_size=tuple(cfg.input_size),      # (X, Y, Z)
        patch_size=tuple(cfg.patch_size),      # (px, py, pz)
        tubelet_size=cfg.tubelet_size,           # frames per token
        num_frames=cfg.initial_step,           # T = context length
        in_chans=cfg.in_chans,         # 4  (u, v, w, smoke)
        encoder_embed_dim=cfg.encoder_embed_dim,
        decoder_embed_dim=cfg.decoder_embed_dim,
        depth=cfg.depth,        # encoder depth
        num_heads=cfg.num_heads,    # encoder heads
        drop_path_rate=cfg.drop_path_rate,
    ).to(dev)

    model = BatchFirstWrapper(vit)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    suffix = "_".join(str(x) for x in cfg.train_subsample)
    model_path = f"{cfg.model_name}_ds{suffix}.pt"

    optim = create_optimizer_v2(
        model,
        opt='adamw',
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        layer_decay=cfg.layer_decay,)

    iters_per_epoch = len(train_loader)
    total_iters = cfg.epochs * iters_per_epoch
    warmup_iters = int(0.05 * total_iters)

    warmup_sched = LinearLR(
        optim,
        start_factor=1e-8,             # tiny LR
        end_factor=1.0,              # reaches base LR
        total_iters=warmup_iters,
    )

    cosine_sched = CosineAnnealingLR(
        optim,
        T_max=max(1, total_iters - warmup_iters),
        eta_min=1e-7,               # final LR ~0
    )

    scheduler = SequentialLR(
        optim,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_iters],    # switch at this step count
    )

    wandb.init(project="3D_NS_transformer",
               config=OmegaConf.to_container(cfg))
    wandb.define_metric("sim_hours")
    for k in ["train_primary_loss", "val_primary_loss", "learning rate"]:
        wandb.define_metric(k, step_metric="sim_hours")
    start_time = time.time()
    best_val, start_ep = float("inf"), 0

    if cfg.continue_training and Path(model_path).exists():
        ckpt = torch.load(model_path, map_location=dev)
        model.load_state_dict(ckpt["model_state_dict"])
        optim.load_state_dict(ckpt["optimizer_state_dict"])
        for s in optim.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(dev)
        start_ep, best_val = ckpt["epoch"], ckpt["loss"]

    if not cfg.if_training:
        model.load_state_dict(torch.load(
            model_path, map_location=dev)["model_state_dict"])
        model.eval()
        err_tuple = metrics(
            val_loader, model, Lx=1.0, Ly=1.0, Lz=1.0,
            plot=cfg.plot, channel_plot=cfg.channel_plot, model_name=cfg.model_name,
            x_min=cfg.x_min, x_max=cfg.x_max, y_min=cfg.y_min, y_max=cfg.y_max,
            t_min=cfg.t_min, t_max=cfg.t_max, mode="Transformer",
        )
        names = ["RMSE", "nRMSE", "CSV", "Max", "Boundary", "FourierBands"]
        err = {k: v for k, v in zip(names, err_tuple[:5])}
        err["FourierBands"] = err_tuple[5]
        print("Evaluation metrics:", err)
        wandb.log(err)
        with open(f"{cfg.model_name}_ds{suffix}_eval_metrics.pkl", "wb") as f:
            pickle.dump(err, f)
        return

    accum_iter = cfg.accumulation_steps
    scaler = GradScaler(enabled=True,
                        init_scale=2**8,
                        growth_interval=1000)

    for ep in tqdm(range(start_ep, cfg.epochs), desc="Epoch"):
        model.train()
        tr_loss, nbatches = 0.0, 0
        optim.zero_grad(set_to_none=True)

        for bidx, (x_in, y_tgt) in enumerate(train_loader, start=1):
            x_in = x_in.to(dev, non_blocking=True)
            y_tgt = y_tgt.to(dev, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = nrmse(model(x_in), y_tgt).mean() / accum_iter

            scaler.scale(loss).backward()
            nbatches += 1

            if bidx % accum_iter == 0 or bidx == len(train_loader):
                scaler.unscale_(optim)

                total_gnorm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0)

                bad_step = any(has_inf_or_nan(p.grad) for p in model.parameters()
                               if p.grad is not None)

                if bad_step:
                    print(f"Skipping update: NaN/Inf gradients at "
                          f"epoch {ep}, batch {bidx}")
                    optim.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad(set_to_none=True)

            tr_loss += loss.item() * accum_iter

            if bidx == 1:
                wandb.log({
                    "epoch": ep,
                    "lr": scheduler.get_last_lr()[0],
                    "amp_scale": scaler.get_scale(),
                    "grad_norm": total_gnorm,
                    "train_loss_batch0": loss.item() * accum_iter,
                })

        tr_loss /= nbatches

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_in, y_tgt in val_loader:
                x_in = x_in.to(dev, non_blocking=True)
                y_tgt = y_tgt.to(dev, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    val_loss += nrmse(model(x_in), y_tgt).mean().item()
        val_loss /= len(val_loader)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optim.state_dict(),
                        "loss": best_val}, model_path)

        sim_hours = (time.time()-start_time)/3600
        wandb.log({
            "sim_hours": sim_hours,
            "Epoch": ep,
            "train_primary_loss": tr_loss,
            "val_primary_loss": val_loss,
            "learning rate": scheduler.get_last_lr()[0]})
        print(f"Epoch {ep:03d} — train {tr_loss:.4e} — val {val_loss:.4e}")

        gc.collect()
        torch.cuda.empty_cache()

# Hydra entry


@hydra.main(version_base=None,
            config_path=".",
            config_name="config_transformer_ns_3d")
def _main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    run_training(cfg.args if "args" in cfg else cfg)


if __name__ == "__main__":
    _main()
