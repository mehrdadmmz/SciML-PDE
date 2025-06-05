# train_transformer_ns.py  –  batch-first, multi-GPU, AMP, grad-accum
from __future__ import annotations

# ───── stdlib / third-party ─────
import gc
import random
import pickle
import time
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

# ───── project-local imports ─────
from utils import TransformerDataset
from transformer import PretrainVisionTransformer
from metrics import metrics

# ───────────────── backend flags ─────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass

# ───────────────── helpers ───────────────────────
# ────────────────────────────────────────────────────────────


def load_weights_flexibly(model: nn.Module, ckpt_path: Path | str, device):
    """
    Load a checkpoint even if its keys are missing 'vit.' or 'module.' prefixes.
    """
    raw = torch.load(ckpt_path, map_location=device)["model_state_dict"]

    # 1) strip eventual 'module.' from DataParallel checkpoints
    stripped = {k.replace("module.", ""): v for k, v in raw.items()}

    try:                       # try direct load first
        model.load_state_dict(stripped, strict=True)
        return
    except RuntimeError:
        pass

    # 2) if we’re inside BatchFirstWrapper, add 'vit.' prefix and try again
    if isinstance(model, BatchFirstWrapper):
        prefixed = {("vit." + k if not k.startswith("vit.") else k): v
                    for k, v in stripped.items()}
        # allow heads etc. to differ
        model.load_state_dict(prefixed, strict=False)
    else:
        model.load_state_dict(stripped, strict=False)
# ────────────────────────────────────────────────────────────


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def nrmse(pred: torch.Tensor, tgt: torch.Tensor):
    spatial = (1, 2, 3)
    tgt_norm = tgt.pow(2).mean(spatial, keepdim=True) + 1e-7
    return (pred - tgt).pow(2).mean(spatial, keepdim=True) / tgt_norm


# ───────────────── wrapper (batch-first → time-first) ─────
class BatchFirstWrapper(nn.Module):
    """Accept (B,T,C,H,W) and permute to (T,B,…) expected by ViT"""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x_btchw):
        x_tbchw = x_btchw.permute(1, 0, 2, 3, 4)
        return self.vit(x_tbchw)


# ───────────────────────────────────────────────────────────
def run_training(cfg: DictConfig):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # ───── datasets / loaders (train & quick val) ───────────
    train_ds = TransformerDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        train_subsample=cfg.train_subsample,
        if_test=False,
        rollout_test=1,               # always 1-step during training
    )
    val_ds = TransformerDataset(
        initial_step=cfg.initial_step,
        saved_folder=cfg.base_path,
        if_test=True,
        rollout_test=1,
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
    print(
        f"train batches: {len(train_loader)} — val batches: {len(val_loader)}")

    # ───── model ────────────────────────────────────────────
    vit = PretrainVisionTransformer(
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

    model = vit

    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
    #     model = torch.nn.DataParallel(model)

    # ───── paths ────────────────────────────────────────────
    suffix = "_".join(str(x) for x in cfg.train_subsample)
    model_path = f"{cfg.model_name}_ds{suffix}.pt"

    # ───── optimiser / scheduler ────────────────────────────
    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, cfg.epochs * len(train_loader)
        )
        if cfg.scheduler == "cosine"
        else torch.optim.lr_scheduler.StepLR(optim, cfg.scheduler_step, cfg.scheduler_gamma)
    )

    # ───── logging ─────────────────────────────────────────
    wandb.init(project="2D_NS_transformer", config=OmegaConf.to_container(cfg))
    start_time = time.time()
    wandb.define_metric("sim_hours")
    wandb.define_metric("train_primary_loss", step_metric="sim_hours")
    wandb.define_metric("val_primary_loss", step_metric="sim_hours")
    wandb.define_metric("learning rate", step_metric="sim_hours")

    best_val, start_ep = float("inf"), 0

    # ───── optional checkpoint reload ──────────────────────
    if cfg.continue_training and Path(model_path).exists():
        load_weights_flexibly(model, model_path, dev)
        optim.load_state_dict(torch.load(model_path, map_location=dev)[
                              "optimizer_state_dict"])
        for s in optim.state.values():
            for k, v in s.items():
                if isinstance(v, torch.Tensor):
                    s[k] = v.to(dev)
        ckpt = torch.load(model_path, map_location=dev)
        start_ep, best_val = ckpt["epoch"], ckpt["loss"]

    # ───── evaluation-only (single or rollout) ─────────────
    if not cfg.if_training:
        # reload best weights
        load_weights_flexibly(model, model_path, dev)
        model.eval()

        # if doing multi-step rollout, rebuild the loader
        if cfg.val_type == "rollout" and cfg.rollout_test > 1:
            val_ds = TransformerDataset(
                initial_step=cfg.initial_step,
                saved_folder=cfg.base_path,
                if_test=True,
                rollout_test=cfg.rollout_test,
            )
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=True,
                persistent_workers=True,
            )

        err_tuple = metrics(
            val_loader,
            model,
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            plot=cfg.plot,
            channel_plot=cfg.channel_plot,
            model_name=cfg.model_name,
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            y_min=cfg.y_min,
            y_max=cfg.y_max,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            mode="Transformer",
            val_type=cfg.val_type,
            rollout_test=cfg.rollout_test,
            initial_step=cfg.initial_step,
        )

        names = ["RMSE", "nRMSE", "CSV", "Max", "Boundary", "FourierBands"]
        err = {k: v for k, v in zip(names, err_tuple[:5])}
        err["FourierBands"] = err_tuple[5]
        print("Evaluation metrics:", err)
        wandb.log(err)
        with open(f"{cfg.model_name}_ds{suffix}_eval_metrics.pkl", "wb") as f:
            pickle.dump(err, f)
        return

    # ───── training loop ───────────────────────────────────
    accum_iter = 2
    scaler = torch.cuda.amp.GradScaler()

    for ep in tqdm(range(start_ep, cfg.epochs), desc="Epoch"):
        model.train()
        tr_loss = 0.0
        nbatches = 0
        optim.zero_grad(set_to_none=True)

        for bidx, (x_in, y_tgt) in enumerate(train_loader):
            x_in = x_in.to(dev, non_blocking=True)
            y_tgt = y_tgt.to(dev, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                y_pred = model(x_in)
                loss = nrmse(y_pred, y_tgt).mean() / accum_iter

            scaler.scale(loss).backward()
            nbatches += 1

            if (bidx + 1) % accum_iter == 0 or (bidx + 1) == len(train_loader):
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            tr_loss += loss.item() * accum_iter  # restore scale

        tr_loss /= nbatches

        # ───── quick 1-step validation ──────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_in, y_tgt in val_loader:
                x_in = x_in.to(dev, non_blocking=True)
                y_tgt = y_tgt.to(dev, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    val_loss += nrmse(model(x_in), y_tgt).mean().item()
        val_loss /= len(val_loader)

        # ───── checkpoint ──────────────────────────────────
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

        # ───── logging ─────────────────────────────────────
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
        print(f"Epoch {ep:03d} — train {tr_loss:.4e} — val {val_loss:.4e}")

        gc.collect()
        torch.cuda.empty_cache()


# ───────────────── Hydra entry ─────────────────────────────
@hydra.main(version_base=None, config_path=".", config_name="config_transformer_ns")
def _main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    run_training(cfg.args if "args" in cfg else cfg)


if __name__ == "__main__":
    _main()
