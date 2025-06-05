# utils.py   (dataset section only)

from __future__ import annotations
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ReactionDiffusionTransformerDataset(Dataset):

    def __init__(
        self,
        initial_step: int = 10,
        saved_folder: str = "../data_gen/data/",
        if_test: bool = False,
        test_ratio: float = 0.1,
        train_subsample: list[int | float] = [900, 900, 900],
        rollout_steps: int = 1,
    ):

        self.file_path = Path(
            saved_folder, "2D_diff-react_test_all.h5").resolve()
        self.initial_step = initial_step
        self.if_test = if_test
        self.rollout_steps = rollout_steps

        # list all seed keys once
        with h5py.File(self.file_path, "r") as h5f:
            all_keys = sorted(h5f.keys())

        split = int(len(all_keys) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(all_keys[split:])
        else:
            max_train = train_subsample[0]
            if isinstance(max_train, float) and 0 < max_train < 1:
                max_train = int(split * max_train)
            self.data_list = np.array(all_keys[:max_train])

        self.indices: list[tuple[int, int]] = []
        for seed_idx, key in enumerate(self.data_list):
            with h5py.File(self.file_path, "r") as f:
                T_total = f[key]["data"].shape[0]

            if if_test:
                # one window per seed, starting at t = 0
                self.indices.append((seed_idx, 0))
            else:
                # need room for context AND rollout
                Tmax = (
                    T_total
                    - initial_step
                    - rollout_steps           # >>> CHANGED
                    + 1
                )
                for t_start in range(Tmax):
                    self.indices.append((seed_idx, t_start))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        seed_idx, t_start = self.indices[idx]
        seed_key = self.data_list[seed_idx]

        with h5py.File(self.file_path, "r") as h5f:
            grp = h5f[seed_key]
            data = torch.as_tensor(grp["data"][...], dtype=torch.float32)
            x = torch.as_tensor(grp["grid"]["x"][...], dtype=torch.float32)
            y = torch.as_tensor(grp["grid"]["y"][...], dtype=torch.float32)

        frames = data.permute(0, 3, 1, 2).contiguous()

        # context & rollout targets
        x_in = frames[t_start: t_start + self.initial_step]
        y_roll = frames[
            t_start + self.initial_step:
            t_start + self.initial_step + self.rollout_steps
        ]

        # keep legacy shape for one-step models
        if self.rollout_steps == 1:
            y_roll = y_roll[0]                      # [C, H, W]

        # spatial grid as mesh
        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack((X, Y), dim=-1)          # [H, W, 2]

        return x_in, y_roll, grid


class ReactionDiffusionAuxDataset(torch.utils.data.Dataset):

    def __init__(self, *args, rollout_steps: int = 1, **kwargs):
        # —— discard legacy keywords the base class doesn't need ——
        for _k in ("primary_filename", "auxiliary_filename", "num_aux_samples"):
            kwargs.pop(_k, None)

        # ReactionDiffusionTransformerDataset handles the rest
        self._base = ReactionDiffusionTransformerDataset(
            *args, rollout_steps=rollout_steps, **kwargs
        )
        self.rollout_steps = rollout_steps

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        # base returns: x_in [T,C,H,W], y_roll [R,C,H,W] or [C,H,W], grid [H,W,2]
        x_in, y_roll, grid = self._base[idx]

        # shape bookkeeping
        T, C, H, W = x_in.shape            # context length, channels, spatial
        # 0 or rollout_steps
        R_dim = 0 if y_roll.ndim == 3 else y_roll.shape[0]

        # fabricate 1 auxiliary sample (all zeros) with correct shape
        # xx_aux : [N_aux(=1), T,  C, H, W]
        # yy_aux : [N_aux(=1), R,  C, H, W]   (R collapses to 1 when rollout_steps==1)
        xx_aux = torch.zeros(1, T,  C, H, W, dtype=x_in.dtype)
        if R_dim == 0:          # one-step legacy case
            yy_aux = torch.zeros(1, 1, C, H, W, dtype=x_in.dtype)
        else:
            yy_aux = torch.zeros(1, R_dim, C, H, W, dtype=x_in.dtype)

        grid_aux = grid.clone()            # same spatial grid

        # DataLoader will prepend the batch dimension ⇒
        #   xx_aux  -> [B, N_aux, T, C, H, W]   as the training script expects
        return x_in, y_roll, xx_aux, yy_aux, grid, grid_aux
