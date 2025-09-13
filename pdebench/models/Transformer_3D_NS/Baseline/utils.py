# utils.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TransformerDataset(Dataset):

    def __init__(
        self,
        initial_step: int = 10,
        # 3‑D dataset directory -- to be changed based on the google cloud bucket
        saved_folder: str = "/local-scratch1/3D_NS",
        if_test: bool = False,
        train_subsample: List[int | float] = [900, 900, 900],
    ):
        super().__init__()

        self.initial_step = initial_step
        self.if_test = if_test
        self.folder = Path(saved_folder).resolve()

        if if_test:
            # 25 held‑out seeds
            seed_ids = range(175, 200)
        else:
            if isinstance(train_subsample[0], float) and 0 < train_subsample[0] < 1:
                max_seed = int(900 * train_subsample[0])
                seed_ids = range(max_seed)
            else:
                seed_ids = range(int(train_subsample[0]))

        self.vel_files = [self.folder /
                          f"v_trj_seed{i}_interp.h5" for i in seed_ids]
        self.smoke_files = [self.folder /
                            f"s_trj_seed{i}_interp.h5" for i in seed_ids]

        self.indices: List[Tuple[int, int]] = []
        for file_idx, vel_path in enumerate(self.vel_files):
            with h5py.File(vel_path, "r") as f:
                T_total = f["data"].shape[3]  # (X, Y, Z, T, 3)

            if if_test:
                # One prediction per trajectory (start at t0 = 0)
                self.indices.append((file_idx, 0))
            else:
                for t0 in range(T_total - initial_step):
                    self.indices.append((file_idx, t0))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        file_idx, t0 = self.indices[idx]
        vel_path = self.vel_files[file_idx]
        smoke_path = self.smoke_files[file_idx]

        with h5py.File(vel_path, "r") as hf_vel, h5py.File(smoke_path, "r") as hf_s:
            # Velocity slice:  (X, Y, Z, T_slice, 3)
            v_raw = hf_vel["data"][..., t0: t0 + self.initial_step + 1, :]
            # Smoke slice:     (T_slice, X, Y, Z)
            s_raw = hf_s["data"][t0: t0 + self.initial_step + 1, ...]

        v_trj = np.transpose(v_raw, (3, 0, 1, 2, 4))
        # -> (T, X, Y, Z, 1)
        s_trj = s_raw[..., np.newaxis]
        # -> (T, X, Y, Z, 4)
        frames = np.concatenate([v_trj, s_trj], axis=-1)

        frames = torch.as_tensor(
            frames, dtype=torch.float32)  # (T, X, Y, Z, 4)
        frames = frames.permute(0, 4, 1, 2, 3)                # (T, C, X, Y, Z)

        x_in = frames[: self.initial_step]                  # (T, C, X, Y, Z)
        y_next = frames[self.initial_step]                    # (C, X, Y, Z)

        return x_in, y_next
