from __future__ import annotations


import math as mt
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["ReactionDiffusionDownsampleDataset"]


class ReactionDiffusionDownsampleDataset(Dataset):

    def _resolve_path(self, name: str | Path, folder: Path) -> Path:
        """Return an absolute ``Path`` object for *name* inside *folder*."""
        p = Path(name)
        if not p.suffix:  # add .h5 if user supplied bare stem
            p = p.with_suffix(".h5")
        if not p.is_absolute():
            p = folder / p
        return p.resolve()

    def __init__(
        self,
        primary_filename: str = "2D_diff-react_test_all",
        auxiliary_filename: str = "2D_diff-react_test_diff",
        *,
        downsample_filename: str = "2D_diff-react_decomp_downsample",
        initial_step: int = 10,
        saved_folder: str | Path = "../data_gen/data/",
        if_downsample: bool = False,
        if_test: bool = False,
        test_ratio: float = 0.1,
        train_subsample: List[int | float] = [900, 900, 900],
        num_aux_samples: int = 3,
    ) -> None:
        super().__init__()

        saved_folder = Path(saved_folder).expanduser().resolve()

        self.primary_path: Path = self._resolve_path(
            primary_filename, saved_folder)
        if if_downsample:
            self.aux_path: Path = self._resolve_path(
                downsample_filename, saved_folder)
        else:
            self.aux_path: Path = self._resolve_path(
                auxiliary_filename, saved_folder)

        self.if_downsample: bool = bool(if_downsample)
        self.if_test: bool = bool(if_test)
        self.initial_step: int = int(initial_step)
        self.num_aux_samples: int = 1 if if_test else int(num_aux_samples)

        with h5py.File(self.primary_path, "r") as h5_p, h5py.File(self.aux_path, "r") as h5_a:
            primary_keys = sorted(h5_p.keys())
            aux_keys = sorted(h5_a.keys())

        split_idx = int(len(primary_keys) * (1 - test_ratio))

        if self.if_test:
            self.primary_list = np.array(primary_keys[split_idx:])
            self.aux_list = np.array(aux_keys[split_idx:])
        else:
            # Honour train_subsample counts (identical semantics to FNO)
            max_primary = train_subsample[1]
            max_aux = train_subsample[2]
            if isinstance(max_primary, float) and 0 < max_primary < 1:
                max_primary = int(split_idx * max_primary)
            if isinstance(max_aux, float) and 0 < max_aux < 1:
                max_aux = int(split_idx * max_aux)
            self.primary_list = np.array(primary_keys[: max_primary])
            self.aux_list = np.array(aux_keys[: max_aux])

        self.indices: List[Tuple[int, int]] = []
        if self.if_test:
            for i, _ in enumerate(self.primary_list):
                self.indices.append((i, 0))
        else:
            for i, key in enumerate(self.primary_list):
                with h5py.File(self.primary_path, "r") as f:
                    T_total = f[key]["data"].shape[0]
                for t in range(T_total - self.initial_step):
                    self.indices.append((i, t))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        """Load a single training / test sample."""
        primary_idx, t_start = self.indices[idx]

        with h5py.File(self.primary_path, "r") as h5_p, h5py.File(self.aux_path, "r") as h5_a:

            grp_p = h5_p[self.primary_list[primary_idx]]
            data_p = torch.as_tensor(
                grp_p["data"][...], dtype=torch.float32)  # [T, X, Y, V]
            # → [T, C, H, W]
            frames_p = data_p.permute(0, 3, 1, 2).contiguous()

            x_primary = frames_p[t_start: t_start +
                                 self.initial_step]        # [T, C, H, W]
            # [C, H, W]
            y_primary = frames_p[t_start + self.initial_step]

            T_full, _, H_full, W_full = frames_p.shape

            aux_slices: List[torch.Tensor] = []
            aux_targets: List[torch.Tensor] = []

            def _load_aux(aux_key: str) -> torch.Tensor:

                grp_a = h5_a[aux_key]
                data_a = torch.as_tensor(
                    # [T, X_d, Y_d, V]
                    grp_a["data"][...], dtype=torch.float32)
                # → [T, C, H_d, W_d]
                frames_a = data_a.permute(0, 3, 1, 2).contiguous()

                if self.if_downsample:
                    # Interpolate from coarse grid/time to primary resolution
                    frames_tf = frames_a.permute(1, 0, 2, 3).unsqueeze(
                        0)         # [1, C, T_d, H_d, W_d]
                    frames_tf = F.interpolate(
                        frames_tf,
                        size=(T_full, H_full, W_full),
                        mode="trilinear",
                        align_corners=False,
                    )
                    frames_a = frames_tf.squeeze(0).permute(
                        1, 0, 2, 3).contiguous()  # [T, C, H, W]

                return frames_a

            if self.if_test:
                aux_key = self.aux_list[idx]  # positional match, same as FNO
                frames_a = _load_aux(aux_key)
                aux_slices.append(
                    frames_a[t_start: t_start + self.initial_step])
                aux_targets.append(frames_a[t_start + self.initial_step])
            else:
                for i in range(self.num_aux_samples):
                    aux_idx = primary_idx * self.num_aux_samples + i
                    aux_idx %= len(self.aux_list)  # safety wrap‑around
                    frames_a = _load_aux(self.aux_list[aux_idx])
                    aux_slices.append(
                        frames_a[t_start: t_start + self.initial_step])
                    aux_targets.append(frames_a[t_start + self.initial_step])

            x_aux = torch.stack(aux_slices)      # [N_aux, T, C, H, W]
            y_aux = torch.stack(aux_targets)     # [N_aux, C, H, W]

            x_vec = torch.as_tensor(
                grp_p["grid"]["x"][...], dtype=torch.float32)
            y_vec = torch.as_tensor(
                grp_p["grid"]["y"][...], dtype=torch.float32)
            X, Y = torch.meshgrid(x_vec, y_vec, indexing="ij")
            grid = torch.stack((X, Y), dim=-1)   # [H, W, 2]
            grid_aux = grid                      # identical grid for aux

        return x_primary, y_primary, x_aux, y_aux, grid, grid_aux
