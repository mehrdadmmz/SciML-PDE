from __future__ import annotations

import math as mt
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["TransformerDatasetAux"]


class TransformerDatasetAux(Dataset):

    def __init__(
        self,
        *,
        initial_step: int = 10,
        saved_folder: str | Path = "/local-scratch1/3D_NS_interp",
        aux_saved_folder: str | Path = "/local-scratch2/3D_NS_comp_interp",
        if_test: bool = False,
        train_subsample: List[int | float] = [900, 900, 900],
        num_aux_samples: int = 3,
    ) -> None:
        super().__init__()

        self.initial_step = int(initial_step)
        self.if_test = bool(if_test)
        self.num_aux_samples = int(num_aux_samples)

        # Resolve directories
        self.folder = Path(saved_folder).expanduser().resolve()
        self.aux_folder = Path(aux_saved_folder).expanduser().resolve()

        if self.if_test:
            # 25 held‑out seeds → 275 … 199  (matching FNO‑aux)
            primary_ids = range(275, 300)
            aux_ids = range(275, 300)
        else:
            n_primary = int(train_subsample[1])
            n_aux = int(train_subsample[2])
            primary_ids = range(n_primary)
            aux_ids = range(n_aux)

        def _vel_path(i: int, root: Path) -> Path:
            return root / f"v_trj_seed{i}_interp.h5"

        def _smoke_path(i: int, root: Path) -> Path:
            return root / f"s_trj_seed{i}_interp.h5"

        self.primary_vel = [_vel_path(i, self.folder) for i in primary_ids]
        self.primary_smoke = [_smoke_path(i, self.folder) for i in primary_ids]
        self.aux_vel = [_vel_path(i, self.aux_folder) for i in aux_ids]
        self.aux_smoke = [_smoke_path(i, self.aux_folder) for i in aux_ids]

        self.indices: List[Tuple[int, int]] = []
        for f_idx, vel_path in enumerate(self.primary_vel):
            with h5py.File(vel_path, "r") as hf:
                T_total = hf["data"].shape[3]  # (X, Y, Z, T, 3)
            if self.if_test:
                # One window at t0 = 0 per trajectory
                self.indices.append((f_idx, 0))
            else:
                # Every admissible window [t0, t0+T]
                for t0 in range(T_total - self.initial_step):
                    self.indices.append((f_idx, t0))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):  # type: ignore[override]
        primary_idx, t0 = self.indices[idx]

        frames_primary = self._load_window(
            self.primary_vel[primary_idx],
            self.primary_smoke[primary_idx],
            t0,
        )  # (T+1, C, X, Y, Z)
        # (T, C, X, Y, Z)
        primary_x = frames_primary[: self.initial_step]
        primary_y = frames_primary[self.initial_step]          # (C, X, Y, Z)

        if self.if_test:
            aux_vel_paths = [self.aux_vel[primary_idx]]
            aux_smk_paths = [self.aux_smoke[primary_idx]]
        else:
            start = primary_idx * self.num_aux_samples
            end = start + self.num_aux_samples
            aux_vel_paths = self.aux_vel[start:end]
            aux_smk_paths = self.aux_smoke[start:end]

        aux_x_list: List[torch.Tensor] = []
        aux_y_list: List[torch.Tensor] = []
        for v_path, s_path in zip(aux_vel_paths, aux_smk_paths):
            frames_aux = self._load_window(
                v_path, s_path, t0)  # (T+1, C, X, Y, Z)
            aux_x_list.append(frames_aux[: self.initial_step])
            aux_y_list.append(frames_aux[self.initial_step])

        aux_x = torch.stack(aux_x_list, dim=0)  # (Nₐ, T, C, X, Y, Z)
        aux_y = torch.stack(aux_y_list, dim=0)  # (Nₐ, C, X, Y, Z)

        _, _, X, Y, Z = frames_primary.shape
        grid = self._make_grid(X, Y, Z)  # (X, Y, Z, 3)

        return primary_x, primary_y, aux_x, aux_y, grid

    def _load_window(self, vel_path: Path, smoke_path: Path, t0: int) -> torch.Tensor:
        """Load *inclusive* slice [t0, t0+initial_step] and return
        tensor shaped (T+1, C, X, Y, Z) where C = 4 (u,v,w,smoke).
        """
        with h5py.File(vel_path, "r") as hf_v, h5py.File(smoke_path, "r") as hf_s:
            # (X, Y, Z, T_slice, 3)
            v_raw = hf_v["data"][..., t0: t0 + self.initial_step + 1, :]
            # (T_slice, X, Y, Z)
            s_raw = hf_s["data"][t0: t0 + self.initial_step + 1, ...]

        # Transfer & reorder → (T, X, Y, Z, 3/1)
        v_trj = np.transpose(v_raw, (3, 0, 1, 2, 4))          # (T, X, Y, Z, 3)
        s_trj = s_raw[..., np.newaxis]                        # (T, X, Y, Z, 1)
        frames = np.concatenate([v_trj, s_trj], axis=-1)      # (T, X, Y, Z, 4)

        # Torch & final layout → (T, C, X, Y, Z)
        frames_t = torch.as_tensor(
            frames, dtype=torch.float32).permute(0, 4, 1, 2, 3)
        return frames_t  # (T, C, X, Y, Z)

    @staticmethod
    def _make_grid(X: int, Y: int, Z: int) -> torch.Tensor:
        """Create normalised spatial grid (X, Y, Z, 3)."""
        x_lin = torch.linspace(0, 1, X)
        y_lin = torch.linspace(0, 1, Y)
        z_lin = torch.linspace(0, 1, Z)
        Xg, Yg, Zg = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
        return torch.stack((Xg, Yg, Zg), dim=-1)  # (X, Y, Z, 3)
