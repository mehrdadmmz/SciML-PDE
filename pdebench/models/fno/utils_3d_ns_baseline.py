from __future__ import annotations

import math as mt
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import pdb


class FNODatasetMult(Dataset):
    def __init__(
        self,
        initial_step=10,
        saved_folder="../data_gen/data/",
        if_test=False,
        test_ratio=0.1,
        train_subsample=[900, 900, 900],
        rollout_test=1,
    ):
        """
        :param folder: Folder containing the HDF5 files.
        :param initial_step: Number of time steps used as initial condition.
        :param if_test: If True, use test portion of dataset.
        :param test_ratio: Fraction of data used for testing.
        """
        self.folder = Path(saved_folder).resolve()
        # self.files = sorted(list(self.folder.glob("*.h5")))
        if if_test:
            self.primary_files_vel = sorted([
                self.folder / f"v_trj_seed{i}_interp.h5"
                for i in range(175, 200)
                # for i in range(100)
            ]) # (X, Y, Z, T, 3) = (100, 100, 178, 150, 3) 
            self.primary_files_smoke = sorted([
                self.folder / f"s_trj_seed{i}_interp.h5"
                for i in range(175, 200)
                # for i in range(100)
            ]) # (T, X, Y, Z) = (150, 100, 100, 178)

        else:
            self.primary_files_vel = sorted([
                self.folder / f"v_trj_seed{i}_interp.h5"
                for i in range(train_subsample[0])
            ])
            self.primary_files_smoke = sorted([
                self.folder / f"s_trj_seed{i}_interp.h5"
                for i in range(train_subsample[0])
            ])
            
        self.initial_step = initial_step
        self.if_test = if_test
        self.rollout_test = rollout_test

        self.indices = []
        for file_idx, vel_file in enumerate(self.primary_files_vel):
            with h5py.File(vel_file, 'r')  as f:
                T = f['data'].shape[3]  # from shape (X, Y, Z, T, 3)
            if self.if_test:
                self.indices.append((file_idx, 0))
            else:
                for t_start in range(T - initial_step):
                    self.indices.append((file_idx, t_start))
        # pdb.set_trace()

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        primary_file_idx, t0 = self.indices[idx]

         # Load primary from HDF5
        vel_path = self.primary_files_vel[primary_file_idx]
        smoke_path = self.primary_files_smoke[primary_file_idx]
        with h5py.File(vel_path, 'r') as hf_vel, h5py.File(smoke_path, 'r') as hf_smoke:
            v_raw = hf_vel['data'][..., t0:(t0 + self.initial_step + self.rollout_test), :]
            s_raw = hf_smoke['data'][t0:(t0 + self.initial_step + self.rollout_test), ...]
            
        v_trj = np.transpose(v_raw, (3, 0, 1, 2, 4))  # (T, X, Y, Z, 3)
        s_trj = s_raw[..., np.newaxis]  # (T, X, Y, Z, 1)
        combined = np.concatenate([v_trj, s_trj], axis=-1)  # (T, X, Y, Z, 4)
        primary_sample = torch.tensor(combined, dtype=torch.float).permute(1, 2, 3, 0, 4)
        primary_slice = primary_sample[..., :self.initial_step, :]
        primary_target = primary_sample[..., self.initial_step:(self.initial_step + self.rollout_test), :]

        dim = len(primary_sample.shape) - 2

        if dim == 2:
            x_dim, y_dim = primary_sample.shape[0], primary_sample.shape[1]
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
            grid = torch.stack((X, Y), dim=-1)  # shape: (x_dim, y_dim, 2)
        elif dim == 3:
            x_dim, y_dim, z_dim = primary_sample.shape[0], primary_sample.shape[1], primary_sample.shape[2]
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            z_lin = torch.linspace(0, 1, z_dim)
            X, Y, Z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
            #pdb.set_trace()
            grid = torch.stack((X, Y, Z), dim=-1)  # shape: (x_dim, y_dim, 3)
            
            
        return primary_slice, primary_target, grid