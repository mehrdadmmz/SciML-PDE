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
        saved_folder="../data_gen/ns_256/",
        if_test=False,
        # test_ratio=0.1,
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
            self.primary_files = sorted([
                self.folder / f"ns_incom_inhom_2d_256-{i}.h5" 
                for i in range(250,275)  
            ])
        else:
            if isinstance(train_subsample[0], float) and train_subsample[0] < 1:
                # When using a fraction, load the first file
                self.primary_files = sorted([self.folder / "ns_incom_inhom_2d_256-0.h5"])
            else:
                self.primary_files = sorted([
                    self.folder / f"ns_incom_inhom_2d_256-{i}.h5" 
                    for i in range(train_subsample[0]) 
                ])
            
            
        self.initial_step = initial_step
        self.if_test = if_test
        self.rollout_test = rollout_test

        # Build a list of all indices.
        # Each index is a tuple: (filename, batch_index, t_start)
        # Preload everything into memory
        self.indices = []
        for file_idx, pth in enumerate(self.primary_files):
            # read shapes
            with h5py.File(pth, 'r') as f:
                B, T = f['velocity'].shape[:2]
            # determine number of trajectories
            if (not if_test) and isinstance(train_subsample[0], float) and train_subsample[0] < 1:
                num_traj = int(B * train_subsample[0])
            else:
                num_traj = B
            # append indices
            for b in range(num_traj):
                if self.if_test:
                    self.indices.append((file_idx, b, 0))
                else:
                    for t0 in range(T - initial_step):
                        self.indices.append((file_idx, b, t0))
        #pdb.set_trace()

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        file_idx, b, t0 = self.indices[idx]
        pth = self.primary_files[file_idx]

        # Load primary on-demand
        with h5py.File(pth, 'r') as f:
            vel = f['velocity'][b, t0 : (t0 + self.initial_step + self.rollout_test), ...]  # includes target
            par = f['particles'][b, t0 : (t0 + self.initial_step + self.rollout_test), ...]
        primary_np = np.concatenate([vel, par], axis=-1)        # one concat of shape [t+1, X, Y, 3]
        primary_np = torch.tensor(primary_np, dtype=torch.float)

        permute_idx = list(range(1, len(primary_np.shape) - 1))
        permute_idx.extend([0, -1]) # [x1, ..., xd, t, v]
        primary_full = primary_np.permute(permute_idx)

        primary_slice = primary_full[..., :self.initial_step, :]
        primary_target = primary_full[..., self.initial_step:(self.initial_step + self.rollout_test), :]

        # Extract spatial dimension of data
        dim = len(primary_full.shape) - 2

        if dim == 2:
            x_dim, y_dim = primary_full.shape[0], primary_full.shape[1]
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
            # pdb.set_trace()
            grid = torch.stack((X, Y), dim=-1)  # shape: (x_dim, y_dim, 2)
        elif dim == 3:
            x_dim, y_dim, z_dim = primary_full.shape[0], primary_full.shape[1], primary_full.shape[2]
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            z_lin = torch.linspace(0, 1, z_dim)
            X, Y, Z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
            grid = torch.stack((X, Y, Z), dim=-1)  # shape: (x_dim, y_dim, 2)
            
            
        return primary_slice, primary_target, grid