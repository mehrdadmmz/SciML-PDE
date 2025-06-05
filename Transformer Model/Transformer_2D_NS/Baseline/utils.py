# utils.py
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TransformerDataset(Dataset):

    def __init__(
        self,
        initial_step: int = 10,
        saved_folder: str = "../data_gen/ns_256/",
        if_test: bool = False,
        train_subsample: list[int | float] = [900, 900, 900],
    ):
        self.initial_step = initial_step
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
                    # one window per trajectory (t0 = 0)
                    self.indices.append((file_idx, b, 0))
                else:
                    # sliding window across time
                    for t0 in range(T - initial_step):
                        self.indices.append((file_idx, b, t0))

    # pytorch dataset interface
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, traj_idx, t0 = self.indices[idx]
        pth = self.primary_files[file_idx]

        # load primary data on demand
        with h5py.File(pth, "r") as f:
            # velocity: (B, T, H, W, 2)   – u, v
            # particles: (B, T, H, W, 1) – p   pressure
            vel = f["velocity"][traj_idx, t0: t0 +
                                self.initial_step + 1]  # (T+1, H, W, 2)
            par = f["particles"][traj_idx, t0: t0 +
                                 self.initial_step + 1]  # (T+1, H, W, 1)

        # stack --> tensor, reshape for ViT input
        frames = np.concatenate([vel, par], axis=-1)  # (T+1, H, W, 3)
        frames = torch.as_tensor(frames, dtype=torch.float32)

        # ViT wants (T, C, H, W); final target frame is timestep T
        frames = frames.permute(0, 3, 1, 2)           # (T+1, 3, H, W)
        x_in = frames[: self.initial_step]            # (T,   3, H, W)
        y_next = frames[self.initial_step]            # (3, H, W)

        return x_in, y_next
