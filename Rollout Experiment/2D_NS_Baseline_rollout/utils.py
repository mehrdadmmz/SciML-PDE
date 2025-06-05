# utils.py   ← just replace the TransformerDataset class definition
from __future__ import annotations
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    """
    Dataset for 256×256 Navier-Stokes movies used by the ViT–based solver.

    Args
    ----
    initial_step : length of the context fed to the model (default = 10)
    rollout_test : number of future steps returned as the label
                   • 1  → single-step training / validation   (shape = C×H×W)
                   • n>1→ multi-step rollout evaluation       (shape = n×C×H×W)

    Each __getitem__ returns
        x_in   : [T, C, H, W]             – frames 0‥T-1
        y_next :  single frame **or**     – when rollout_test == 1
                 [R, C, H, W]             – when rollout_test  > 1
    """

    def __init__(
        self,
        initial_step: int = 10,
        saved_folder: str = "../data_gen/ns_256/",
        if_test: bool = False,
        train_subsample: list[int | float] = [900, 900, 900],
        rollout_test: int = 1,                       # --- NEW
    ):
        self.initial_step = initial_step
        self.rollout_test = rollout_test            # --- NEW
        self.if_test = if_test

        self.folder = Path(saved_folder).resolve()
        if if_test:
            self.primary_files = sorted(
                self.folder / f"ns_incom_inhom_2d_256-{i}.h5" for i in range(250, 275)
            )
        else:
            if isinstance(train_subsample[0], float) and 0 < train_subsample[0] < 1:
                self.primary_files = [
                    self.folder / "ns_incom_inhom_2d_256-0.h5"]
            else:
                self.primary_files = sorted(
                    self.folder / f"ns_incom_inhom_2d_256-{i}.h5"
                    for i in range(train_subsample[0])
                )

        # build list of windows: (file_idx, traj_idx, t0)
        self.indices: list[tuple[int, int, int]] = []
        for file_idx, pth in enumerate(self.primary_files):
            with h5py.File(pth, "r") as f:
                B, T = f["velocity"].shape[:2]

            num_traj = (
                int(B * train_subsample[0])
                if (not if_test and isinstance(train_subsample[0], float) and train_subsample[0] < 1)
                else B
            )

            for b in range(num_traj):
                if if_test:
                    self.indices.append((file_idx, b, 0))
                else:
                    for t0 in range(T - initial_step):
                        self.indices.append((file_idx, b, t0))

    # ── PyTorch hooks ────────────────────────────────────────────
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, traj_idx, t0 = self.indices[idx]
        pth = self.primary_files[file_idx]

        with h5py.File(pth, "r") as f:
            vel = f["velocity"][traj_idx, t0: t0 +
                                self.initial_step + self.rollout_test]
            par = f["particles"][traj_idx, t0: t0 +
                                 self.initial_step + self.rollout_test]

        frames = np.concatenate([vel, par], axis=-1)        # (T+R, H, W, 3)
        frames = torch.as_tensor(
            frames, dtype=torch.float32).permute(0, 3, 1, 2)

        # (T,   C, H, W)
        x_in = frames[: self.initial_step]
        y_next = frames[self.initial_step: self.initial_step +
                        self.rollout_test]

        # keep training shape identical to the old loader
        if self.rollout_test == 1:
            y_next = y_next.squeeze(0)                              # (C, H, W)

        return x_in, y_next
