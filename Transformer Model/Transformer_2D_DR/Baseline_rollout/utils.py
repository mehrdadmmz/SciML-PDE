# utils.py

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
        rollout_steps: int = 1,                     # >>> NEW
    ):

        self.file_path = Path(
            saved_folder, "2D_diff-react_test_all.h5").resolve()
        self.initial_step = initial_step
        self.if_test = if_test
        self.rollout_steps = rollout_steps         # >>> NEW

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
                self.indices.append((seed_idx, 0))
            else:
                # must leave enough room for context *and* rollout
                Tmax = T_total - initial_step - rollout_steps + 1     # >>> CHANGED
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

        x_in = frames[t_start: t_start + self.initial_step]
        y_roll = frames[t_start + self.initial_step:
                        t_start + self.initial_step + self.rollout_steps]

        if self.rollout_steps == 1:
            y_roll = y_roll[0]               # → [C, H, W]

        X, Y = torch.meshgrid(x, y, indexing="ij")
        grid = torch.stack((X, Y), dim=-1)   # [H, W, 2]

        return x_in, y_roll, grid
