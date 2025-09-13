from __future__ import annotations


import math as mt
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ReactionDiffusionAuxDataset"]


class ReactionDiffusionAuxDataset(Dataset):

    def __init__(
        self,
        primary_filename: str,
        auxiliary_filename: str,
        *,
        initial_step: int = 10,
        saved_folder: str = "../data_gen/data/",
        if_test: bool = False,
        test_ratio: float = 0.1,
        train_subsample: List[int | float] = [900, 900, 900],
        num_aux_samples: int = 3,
    ) -> None:
        super().__init__()

        self.primary_path = Path(
            saved_folder, f"{primary_filename}.h5").resolve()
        self.aux_path = Path(
            saved_folder, f"{auxiliary_filename}.h5").resolve()

        self.initial_step = initial_step
        self.if_test = if_test
        self.num_aux_samples = 1 if if_test else num_aux_samples

        with h5py.File(self.primary_path, "r") as h5_p, h5py.File(self.aux_path, "r") as h5_a:
            primary_keys = sorted(h5_p.keys())
            aux_keys = sorted(h5_a.keys())

        split_idx = int(len(primary_keys) * (1 - test_ratio))
        if if_test:
            self.primary_list = np.array(primary_keys[split_idx:])
            self.aux_list = np.array(aux_keys[split_idx:])
        else:
            # Honour train_subsample counts (same semantics as FNO file)
            max_primary = train_subsample[1]
            max_aux = train_subsample[2]
            if isinstance(max_primary, float) and 0 < max_primary < 1:
                max_primary = int(split_idx * max_primary)
            if isinstance(max_aux, float) and 0 < max_aux < 1:
                max_aux = int(split_idx * max_aux)
            self.primary_list = np.array(primary_keys[: max_primary])
            self.aux_list = np.array(aux_keys[: max_aux])

        self.indices: List[Tuple[int, int]] = []
        if if_test:
            for i, _ in enumerate(self.primary_list):
                self.indices.append((i, 0))
        else:
            for i, key in enumerate(self.primary_list):
                with h5py.File(self.primary_path, "r") as f:
                    T_total = f[key]["data"].shape[0]
                for t in range(T_total - initial_step):
                    self.indices.append((i, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):  # noqa: D401
        primary_idx, t_start = self.indices[idx]

        with h5py.File(self.primary_path, "r") as h5_p, h5py.File(self.aux_path, "r") as h5_a:

            grp_p = h5_p[self.primary_list[primary_idx]]
            data_p = torch.as_tensor(
                grp_p["data"][...], dtype=torch.float32)  # [T, X, Y, V]
            frames_p = data_p.permute(
                0, 3, 1, 2).contiguous()  # → [T, V, H, W]
            x_primary = frames_p[t_start: t_start +
                                 self.initial_step]        # [T, C, H, W]
            # [C, H, W]
            y_primary = frames_p[t_start + self.initial_step]

            aux_slices: List[torch.Tensor] = []
            aux_targets: List[torch.Tensor] = []

            if self.if_test:
                aux_key = self.aux_list[idx]  # same positional index
                grp_a = h5_a[aux_key]
                data_a = torch.as_tensor(
                    grp_a["data"][...], dtype=torch.float32)
                frames_a = data_a.permute(0, 3, 1, 2).contiguous()
                aux_slices.append(
                    frames_a[t_start: t_start + self.initial_step])
                aux_targets.append(frames_a[t_start + self.initial_step])
            else:
                for i in range(self.num_aux_samples):
                    aux_idx = primary_idx * self.num_aux_samples + i
                    aux_idx %= len(self.aux_list)  # wrap‑around safety
                    grp_a = h5_a[self.aux_list[aux_idx]]
                    data_a = torch.as_tensor(
                        grp_a["data"][...], dtype=torch.float32)
                    frames_a = data_a.permute(0, 3, 1, 2).contiguous()
                    aux_slices.append(
                        frames_a[t_start: t_start + self.initial_step])
                    aux_targets.append(frames_a[t_start + self.initial_step])

            x_aux = torch.stack(aux_slices)    # [N_aux, T, C, H, W]
            y_aux = torch.stack(aux_targets)   # [N_aux, C, H, W]

            x_vec = torch.as_tensor(
                grp_p["grid"]["x"][...], dtype=torch.float32)
            y_vec = torch.as_tensor(
                grp_p["grid"]["y"][...], dtype=torch.float32)
            X, Y = torch.meshgrid(x_vec, y_vec, indexing="ij")
            grid = torch.stack((X, Y), dim=-1)       # [H, W, 2]
            grid_aux = grid  # identical grid for auxiliary trajectories

        return x_primary, y_primary, x_aux, y_aux, grid, grid_aux
