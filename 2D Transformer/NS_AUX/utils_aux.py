# utils_aux.py
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TransformerAuxDataset(Dataset):
    """
    Navier-Stokes (NS) 256×256 movies with auxiliary streams.

    each __getitem__ returns
    x_primary   : [T,   C, H, W]          (T = initial_step)
    y_primary   : [C,      H, W]

    x_aux       : [N_aux, T, C, H, W]     (N_aux = num_aux_samples or 1 in test)
    y_aux       : [N_aux, C, H, W]
    """

    def __init__(
        self,
        initial_step: int = 10,
        saved_folder: str = "../data_gen/ns_256/",
        aux_saved_folder: str = "../data_gen/basic_eq",
        if_test: bool = False,
        train_subsample: list[int | float] = [900, 900, 900],
        num_aux_samples: int = 24,
    ):
        self.initial_step = initial_step
        self.if_test = if_test
        self.num_aux_samples = num_aux_samples

        self.folder = Path(saved_folder).resolve()
        self.aux_folder = Path(aux_saved_folder).resolve()

        # primary & auxiliary file lists
        if if_test:
            self.primary_files = [
                self.folder / f"ns_incom_inhom_2d_256-{i}.h5" for i in range(250, 275)
            ]
            self.aux_files = [
                self.aux_folder / f"ns_aux_2d_256-{i}.h5" for i in range(250, 275)
            ]
        else:
            # primary
            if isinstance(train_subsample[1], float) and 0 < train_subsample[1] < 1:
                self.primary_files = [
                    self.folder / "ns_incom_inhom_2d_256-0.h5"]
            else:
                self.primary_files = [
                    self.folder / f"ns_incom_inhom_2d_256-{i}.h5"
                    for i in range(train_subsample[1])
                ]
            # auxiliary
            self.aux_files = [
                self.aux_folder / f"ns_aux_2d_256-{i}.h5"
                for i in range(train_subsample[2])
            ]
            # group `num_aux_samples` aux files per primary
            num_primary = len(self.primary_files)
            self.aux_groups = [
                self.aux_files[i * num_aux_samples: (i + 1) * num_aux_samples]
                for i in range(num_primary)
            ]

        # build (file_idx, traj_idx, t0) index list - t0 is t_start
        self.indices: list[tuple[int, int, int]] = []
        for file_index, pth in enumerate(self.primary_files):
            with h5py.File(pth, "r") as f:
                B, T = f["velocity"].shape[:2]

            n_traj = (
                int(B * train_subsample[1])
                if (
                    not if_test
                    and isinstance(train_subsample[1], float)
                    and train_subsample[1] < 1
                )
                else B
            )

            for b in range(n_traj):
                if if_test:
                    self.indices.append((file_index, b, 0))
                else:
                    for t0 in range(T - initial_step):
                        self.indices.append((file_index, b, t0))

    # Dataset interface
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, traj_idx, t0 = self.indices[idx]
        p_primary = self.primary_files[file_idx]

        # primary
        with h5py.File(p_primary, "r") as f:
            vel = f["velocity"][traj_idx, t0: t0 + self.initial_step + 1]
            par = f["particles"][traj_idx, t0: t0 + self.initial_step + 1]

        # (T+1, H, W, 3)
        prim = np.concatenate([vel, par], axis=-1)
        prim = torch.as_tensor(prim, dtype=torch.float32).permute(0, 3, 1, 2)
        # (T,   C, H, W)
        x_prim = prim[: self.initial_step]
        # (C,   H, W)
        y_prim = prim[self.initial_step]

        # auxiliary
        if self.if_test:
            # exactly one aux file matching the primary
            aux_paths = [self.aux_files[file_idx]]
        else:
            aux_paths = self.aux_groups[file_idx]

        x_aux_list, y_aux_list = [], []
        for p_aux in aux_paths:
            with h5py.File(p_aux, "r") as f:
                vel = f["velocity"][traj_idx, t0: t0 + self.initial_step + 1]
                par = f["particles"][traj_idx, t0: t0 + self.initial_step + 1]
            aux = np.concatenate([vel, par], axis=-1)
            aux = torch.as_tensor(aux, dtype=torch.float32).permute(0, 3, 1, 2)
            x_aux_list.append(aux[: self.initial_step])         # (T, C, H, W)
            y_aux_list.append(aux[self.initial_step])           # (C, H, W)

        # (N_aux, T, C, H, W)
        x_aux = torch.stack(x_aux_list, dim=0)
        # (N_aux, C, H, W)
        y_aux = torch.stack(y_aux_list, dim=0)

        return x_prim, y_prim, x_aux, y_aux
