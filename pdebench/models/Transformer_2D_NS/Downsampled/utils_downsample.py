from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TransformerDownsampleDataset(Dataset):

    def __init__(
        self,
        initial_step: int = 10,
        saved_folder: str | Path = "../data_gen/ns_256/",
        aux_saved_folder: str | Path = "../data_gen/basic_eq",
        *,
        if_downsample: bool = False,
        if_test: bool = False,
        train_subsample: List[int | float] | Tuple[int |
                                                   float, ...] = [900, 900, 900],
        num_aux_samples: int = 24,
    ) -> None:
        super().__init__()
        self.initial_step = initial_step
        self.if_downsample = if_downsample
        self.if_test = if_test
        self.num_aux_samples = num_aux_samples

        # Resolve data folders
        self.folder = Path(saved_folder).expanduser().resolve()
        self.aux_folder = Path(aux_saved_folder).expanduser().resolve()

        if self.if_test:

            self.primary_files = [
                self.folder / f"ns_incom_inhom_2d_256-{i}.h5" for i in range(250, 275)
            ]
            self.aux_files = [
                self.aux_folder / f"ns_aux_2d_256-{i}.h5" for i in range(250, 275)
            ]
        else:
            if isinstance(train_subsample[1], float) and 0 < train_subsample[1] < 1:
                # Fractional mode – just the **first** file, subset of trajectories later
                self.primary_files = [
                    self.folder / "ns_incom_inhom_2d_256-0.h5"]
            else:
                # Integer -> explicit number of files
                self.primary_files = [
                    self.folder / f"ns_incom_inhom_2d_256-{i}.h5" for i in range(train_subsample[1])
                ]

            self.aux_files = [
                self.aux_folder / f"ns_aux_2d_256-{i}.h5" for i in range(train_subsample[2])
            ]

            n_primary = len(self.primary_files)
            assert (
                len(self.aux_files) == n_primary * num_aux_samples
            ), "train_subsample[2] must equal n_primary * num_aux_samples"

            self.aux_groups: List[List[Path]] = [
                self.aux_files[i * num_aux_samples: (i + 1) * num_aux_samples]
                for i in range(n_primary)
            ]

        self.indices: List[Tuple[int, int, int]] = []
        for file_idx, pth in enumerate(self.primary_files):
            with h5py.File(pth, "r") as f:
                n_traj, n_time = f["velocity"].shape[:2]

            if (
                not self.if_test
                and isinstance(train_subsample[1], float)
                and train_subsample[1] < 1
            ):
                n_traj = int(n_traj * train_subsample[1])

            for b in range(n_traj):
                if self.if_test:
                    # one fixed window per trajectory
                    self.indices.append((file_idx, b, 0))
                else:
                    # every valid *initial_step*‑length window -> **dense supervision**
                    for t0 in range(n_time - initial_step):
                        self.indices.append((file_idx, b, t0))

    def __len__(self) -> int:  # noqa: D401 (simple‑one‑line docstring)
        """Return the total number of (file, trajectory, window) combos."""

        return len(self.indices)

    def __getitem__(self, idx: int):
        file_idx, traj_idx, t0 = self.indices[idx]
        primary_path = self.primary_files[file_idx]

        with h5py.File(primary_path, "r") as f:
            vel = f["velocity"][traj_idx, t0: t0 + self.initial_step + 1]
            par = f["particles"][traj_idx, t0: t0 + self.initial_step + 1]

        primary_np = np.concatenate([vel, par], axis=-1)
        primary_tf = torch.tensor(primary_np, dtype=torch.float32)

        permute_idx = list(range(1, primary_tf.ndim - 1)) + [0, -1]
        primary_full = primary_tf.permute(*permute_idx)

        # (X,Y,T,V)
        primary_slice = primary_full[..., : self.initial_step, :]
        primary_target = primary_full[...,
                                      self.initial_step: self.initial_step + 1, :]

        primary_slice = primary_slice.permute(
            2, 3, 0, 1)        # CHANGED → (T,V,X,Y)
        primary_target = primary_target.squeeze(
            2).permute(2, 0, 1)  # CHANGED → (V,X,Y)

        if self.if_test:
            aux_paths = [self.aux_files[file_idx]]
        else:
            aux_paths = self.aux_groups[file_idx]

        aux_slice_list, aux_target_list = [], []
        for aux_p in aux_paths:
            with h5py.File(aux_p, "r") as f:
                vel = f["velocity"][traj_idx, t0: t0 + self.initial_step + 1]
                par = f["particles"][traj_idx, t0: t0 + self.initial_step + 1]
            aux_np = np.concatenate([vel, par], axis=-1)
            aux_tf = torch.tensor(aux_np, dtype=torch.float32)
            # (X_aux,Y_aux,T+1,V)
            aux_full = aux_tf.permute(*permute_idx)

            if self.if_downsample:
                x_dim, y_dim = primary_full.shape[:2]
                # x_dim = y_dim = 256
                aux_tmp = aux_full.permute(2, 3, 0, 1)
                aux_tmp = F.interpolate(aux_tmp, size=(x_dim, y_dim),
                                        mode="bilinear", align_corners=False)
                aux_full = aux_tmp.permute(2, 3, 0, 1)

            aux_slice_list.append(aux_full[..., : self.initial_step, :])
            aux_target_list.append(
                aux_full[..., self.initial_step: self.initial_step + 1, :])

        # (N_aux,X,Y,T,V)
        aux_slice = torch.stack(aux_slice_list, dim=0)
        aux_target = torch.stack(
            aux_target_list, dim=0)         # (N_aux,X,Y,1,V)

        aux_slice = aux_slice.permute(0, 3, 4, 1, 2)
        aux_target = aux_target.squeeze(3).permute(
            0, 3, 1, 2)    # CHANGED → (N_aux,V,X,Y)

        dim = primary_full.ndim - 2
        if dim == 2:
            x_dim, y_dim = primary_full.shape[:2]
            x_lin = torch.linspace(0.0, 1.0, x_dim)
            y_lin = torch.linspace(0.0, 1.0, y_dim)
            X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
            grid = torch.stack((X, Y), dim=-1)
            grid_aux = grid
        elif dim == 3:  # pragma: no cover
            x_dim, y_dim, z_dim = primary_full.shape[:3]
            x_lin = torch.linspace(0.0, 1.0, x_dim)
            y_lin = torch.linspace(0.0, 1.0, y_dim)
            z_lin = torch.linspace(0.0, 1.0, z_dim)
            X, Y, Z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
            grid = torch.stack((X, Y, Z), dim=-1)
            grid_aux = grid
        else:
            raise RuntimeError(
                "Unsupported spatial dimensionality: expected 2D or 3D data")

        return (
            primary_slice,    # (T,C,H,W) after DataLoader collation
            primary_target,   # (C,H,W)
            aux_slice,        # (N_aux,T,C,H,W)
            aux_target,       # (N_aux,C,H,W)
            grid,
            grid_aux,
        )
