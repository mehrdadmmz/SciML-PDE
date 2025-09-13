from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class TransformerAuxDataset(Dataset):
    """
    Navier–Stokes movies with auxiliary streams.

    Returns
        x_primary : [T,   C, H, W]
        y_primary : [R,   C, H, W]
        x_aux     : [N_aux, T, C, H, W]
        y_aux     : [N_aux, R, C, H, W]
    """

    def __init__(
        self,
        *,
        initial_step: int = 10,
        saved_folder: str | Path,
        aux_saved_folder: str | Path,
        if_test: bool = False,
        train_subsample: list[int | float] = [900, 900, 900],
        num_aux_samples: int = 24,
        rollout_test: int = 1,
        target_size: int = 256,          # resize everything to this
    ) -> None:
        super().__init__()
        self.initial_step = initial_step
        self.if_test = if_test
        self.rollout_test = rollout_test
        self.num_aux_samples = num_aux_samples
        self.target_size = target_size

        self.folder = Path(saved_folder).resolve()
        self.aux_folder = Path(aux_saved_folder).resolve()

        # ───────── build file lists ────────────────────────────────────
        if if_test:
            self.primary_files = [
                self.folder / f"ns_incom_inhom_2d_256-{i}.h5" for i in range(250, 275)
            ]
            self.aux_files = [
                self.aux_folder / f"ns_aux_2d_256-{i}.h5" for i in range(250, 275)
            ]
        else:
            if isinstance(train_subsample[1], float) and 0 < train_subsample[1] < 1:
                self.primary_files = [
                    self.folder / "ns_incom_inhom_2d_256-0.h5"]
            else:
                self.primary_files = [
                    self.folder / f"ns_incom_inhom_2d_256-{i}.h5"
                    for i in range(train_subsample[1])
                ]
            self.aux_files = [
                self.aux_folder / f"ns_aux_2d_256-{i}.h5"
                for i in range(train_subsample[2])
            ]
            n_primary = len(self.primary_files)
            self.aux_groups = [
                self.aux_files[i * num_aux_samples: (i + 1) * num_aux_samples]
                for i in range(n_primary)
            ]

        # ───────── build (file, traj, t0) indices ──────────────────────
        self.indices: list[tuple[int, int, int]] = []
        for file_idx, pth in enumerate(self.primary_files):
            with h5py.File(pth, "r") as f:
                B, T = f["velocity"].shape[:2]
            n_traj = (
                int(B * train_subsample[1])
                if (
                    not if_test
                    and isinstance(train_subsample[1], float)
                    and 0 < train_subsample[1] < 1
                )
                else B
            )
            for b in range(n_traj):
                if if_test:
                    self.indices.append((file_idx, b, 0))
                else:
                    for t0 in range(T - initial_step - rollout_test + 1):
                        self.indices.append((file_idx, b, t0))

    # ───────────────── helper ──────────────────────────────────────────
    def _resize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Bilinear-resize last two dims to *target_size* (stride-safe)."""
        if tensor.shape[-1] == self.target_size:
            return tensor

        orig_shape = tensor.shape                                 # [..., H, W]
        tensor = tensor.contiguous().reshape(-1, 1,
                                             orig_shape[-2], orig_shape[-1])
        tensor = F.interpolate(tensor,
                               size=self.target_size,
                               mode="bilinear",
                               align_corners=False)
        tensor = tensor.reshape(*orig_shape[:-2],
                                self.target_size, self.target_size)
        return tensor

    # ───────────────── Dataset interface ───────────────────────────────
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        file_idx, traj_idx, t0 = self.indices[idx]
        p_primary = self.primary_files[file_idx]

        rng = slice(t0, t0 + self.initial_step + self.rollout_test)

        # ---------- primary -------------------------------------------
        with h5py.File(p_primary, "r") as f:
            vel = f["velocity"][traj_idx, rng]
            par = f["particles"][traj_idx, rng]
        # (T+R,H,W,3)
        prim = np.concatenate([vel, par], axis=-1)
        prim = torch.as_tensor(prim, dtype=torch.float32).permute(0, 3, 1, 2)

        x_prim = prim[: self.initial_step]                         # [T,C,H,W]
        y_prim = prim[self.initial_step: self.initial_step + self.rollout_test]

        # ---------- auxiliary -----------------------------------------
        aux_paths = (
            [self.aux_files[file_idx]] if self.if_test else self.aux_groups[file_idx]
        )
        x_aux_list, y_aux_list = [], []
        for p_aux in aux_paths:
            with h5py.File(p_aux, "r") as f:
                vel = f["velocity"][traj_idx, rng]
                par = f["particles"][traj_idx, rng]
            aux = torch.as_tensor(np.concatenate([vel, par], axis=-1),
                                  dtype=torch.float32).permute(0, 3, 1, 2)
            x_aux_list.append(aux[: self.initial_step])
            y_aux_list.append(aux[self.initial_step:
                                  self.initial_step + self.rollout_test])

        x_aux = torch.stack(x_aux_list, dim=0)   # [N,T,C,H,W]
        y_aux = torch.stack(y_aux_list, dim=0)   # [N,R,C,H,W]

        # ---------- resize to ViT input size ---------------------------
        x_prim = self._resize(x_prim)
        y_prim = self._resize(y_prim)

        N, T_a, C_a, H, W = x_aux.shape
        x_aux = self._resize(
            x_aux.reshape(N * T_a, C_a, H, W)
        ).reshape(N, T_a, C_a, self.target_size, self.target_size)

        N, R_a, C_a, H, W = y_aux.shape
        y_aux = self._resize(
            y_aux.reshape(N * R_a, C_a, H, W)
        ).reshape(N, R_a, C_a, self.target_size, self.target_size)

        return x_prim, y_prim, x_aux, y_aux
