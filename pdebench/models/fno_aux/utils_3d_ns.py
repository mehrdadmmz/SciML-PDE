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
        aux_saved_folder = "../data_gen/basic_eq",
        if_downsample=False,
        if_test=False,
        # test_ratio=0.1,
        train_subsample=[900, 900, 900],
        num_aux_samples=3,
        rollout_test=1,
    ):
        """
        NS dataset with auxiliary information.
        
        For training:
          - Primary data is loaded from the files in "ns_incom_inhom_2d_256-{i}.h5"
            for i in range(train_subsample[1]).
          - Auxiliary data is loaded from "ns_aux_2d_256-{i}.h5"
            for i in range(train_subsample[2]). Each primary file is paired with
            `num_aux_samples` auxiliary files.
          - For each batch sample in each primary file, all valid time windows of length `initial_step`
            (with the following time step as target) are used.
        
        For testing:
          - Primary data is loaded from files with indices 250 to 274.
          - Auxiliary data is loaded from corresponding auxiliary test files.
          - For each batch sample, a fixed window starting at t=0 is used.
        
        Each NS file is expected to contain:
           - "velocity": shape [B, T, X, Y, 2]
           - "particles": shape [B, T, X, Y, 1]
        These are concatenated along the channel dimension to yield a tensor of shape [B, T, X, Y, 3].
        """

        self.folder = Path(saved_folder).resolve()
        self.aux_folder = Path(aux_saved_folder).resolve()
        self.initial_step = initial_step
        self.if_test = if_test
        self.num_aux_samples = num_aux_samples
        self.rollout_test = rollout_test
       
        if if_test:
            self.primary_files_vel = sorted([
                self.folder / f"v_trj_seed{i}_interp.h5"
                for i in range(175,176)
                # for i in range(175, 200)
            ]) # (X, Y, Z, T, 3) = (100, 100, 178, 150, 3) 
            self.primary_files_smoke = sorted([
                self.folder / f"s_trj_seed{i}_interp.h5"
                for i in range(175,176)
                # for i in range(175, 200)
            ]) # (T, X, Y, Z) = (150, 100, 100, 178)
            self.aux_files_vel = sorted([
                self.aux_folder / f"v_trj_seed{i}.h5"
                # for i in range(275, 300)
                for i in range(275,276)
            ])
            self.aux_files_smoke = sorted([
                self.aux_folder / f"s_trj_seed{i}.h5"
                # for i in range(275, 300)
                for i in range(275,276)
            ])
        else:
            self.primary_files_vel = sorted([
                self.folder / f"v_trj_seed{i}_interp.h5"
                for i in range(train_subsample[1])
            ])
            self.primary_files_smoke = sorted([
                self.folder / f"s_trj_seed{i}_interp.h5"
                for i in range(train_subsample[1])
            ])
            self.aux_files_vel = sorted([
                self.aux_folder / f"v_trj_seed{i}.h5"
                for i in range(train_subsample[2])
            ])
            self.aux_files_smoke = sorted([
                self.aux_folder / f"s_trj_seed{i}.h5"
                for i in range(train_subsample[2])
            ])

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
            v_raw = hf_vel['data'][..., t0:, :]
            s_raw = hf_smoke['data'][t0:, ...]

            # v_raw = hf_vel['data'][..., t0:(t0 + self.initial_step + self.rollout_test), :]
            # s_raw = hf_smoke['data'][t0:(t0 + self.initial_step + self.rollout_test), ...]

        # Rearrange and combine
        v_trj = np.transpose(v_raw, (3, 0, 1, 2, 4))   # (T, X, Y, Z, 3)
        s_trj = s_raw[..., np.newaxis]                # (T, X, Y, Z, 1)
        combined = np.concatenate([v_trj, s_trj], axis=-1)  # (T, X, Y, Z, 4)
        primary_sample = torch.tensor(combined, dtype=torch.float).permute(1, 2, 3, 0, 4)
        primary_slice = primary_sample[..., :self.initial_step, :]
        primary_target = primary_sample

        # Load auxiliary from HDF5
        if self.if_test:
            aux_paths_vel = [self.aux_files_vel[primary_file_idx]]
            aux_paths_smoke = [self.aux_files_smoke[primary_file_idx]]
        else:
            aux_paths_vel = self.aux_files_vel[
                primary_file_idx * self.num_aux_samples:
                primary_file_idx * self.num_aux_samples + self.num_aux_samples
            ]
            aux_paths_smoke = self.aux_files_smoke[
                primary_file_idx * self.num_aux_samples:
                primary_file_idx * self.num_aux_samples + self.num_aux_samples
            ]

        aux_slice_list, aux_target_list = [], []
        for vel_p, smoke_p in zip(aux_paths_vel, aux_paths_smoke):
            with h5py.File(vel_p, 'r') as hf_vel, h5py.File(smoke_p, 'r') as hf_smoke:
                v_raw = hf_vel['data'][..., t0:(t0 + self.initial_step + self.rollout_test), :]
                s_raw = hf_smoke['data'][t0:(t0 + self.initial_step + self.rollout_test), :]

            v_trj = np.transpose(v_raw, (3, 0, 1, 2, 4))
            s_trj = s_raw[..., np.newaxis]
            combined = np.concatenate([v_trj, s_trj], axis=-1)
            aux_sample = torch.tensor(combined, dtype=torch.float).permute(1, 2, 3, 0, 4)
            aux_slice_list.append(aux_sample[..., :self.initial_step, :])
            aux_target_list.append(aux_sample[..., self.initial_step:(self.initial_step + self.rollout_test), :])

        aux_slice = torch.stack(aux_slice_list, dim=0)
        aux_target = torch.stack(aux_target_list, dim=0)


        # Load auxiliary
        # if self.if_test:
        #     v_raw = np.load(self.aux_files_vel[primary_file_idx])['data'][..., t0 : (t0 + self.initial_step + 1), :]
        #     s_raw = np.load(self.aux_files_smoke[primary_file_idx])['data'][t0 : (t0 + self.initial_step + 1), ...]
        #     v_trj = np.transpose(v_raw, (3, 0, 1, 2, 4))
        #     s_trj = s_raw[..., np.newaxis]
        #     combined = np.concatenate([v_trj, s_trj], axis=-1)
        #     aux_sample = torch.tensor(combined, dtype=torch.float).permute(1, 2, 3, 0, 4)
        #     aux_slice = aux_sample[..., :self.initial_step, :].unsqueeze(0)
        #     aux_target = aux_sample[..., self.initial_step:(self.initial_step + 1), :].unsqueeze(0)
        # else:
        #     aux_slice_list = []
        #     aux_target_list = []
        #     for j in range(self.num_aux_samples):
        #         aux_idx = primary_file_idx * self.num_aux_samples + j
        #         v_raw = np.load(self.aux_files_vel[aux_idx])['data.npy'][..., t0 : (t0 + self.initial_step + 1), :]
        #         s_raw = np.load(self.aux_files_smoke[aux_idx])['data.npy'][t0 : (t0 + self.initial_step + 1), ...]
        #         v_trj = np.transpose(v_raw, (3, 0, 1, 2, 4))
        #         s_trj = s_raw[..., np.newaxis]
        #         combined = np.concatenate([v_trj, s_trj], axis=-1)
        #         aux_sample = torch.tensor(combined, dtype=torch.float).permute(1, 2, 3, 0, 4)
        #         aux_slice_list.append(aux_sample[..., :self.initial_step, :])
        #         aux_target_list.append(aux_sample[..., self.initial_step:(self.initial_step + 1), :])
        #     aux_slice = torch.stack(aux_slice_list, dim=0)
        #     aux_target = torch.stack(aux_target_list, dim=0)
        # pdb.set_trace()
        


        # Extract spatial dimension of data
        dim = len(primary_sample.shape) - 2

        if dim == 2:
            x_dim, y_dim = primary_sample.shape[0], primary_sample.shape[1]
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
            grid = torch.stack((X, Y), dim=-1)  # shape: (x_dim, y_dim, 2)
            grid_aux = grid
        elif dim == 3:
            x_dim, y_dim, z_dim = primary_sample.shape[0], primary_sample.shape[1], primary_sample.shape[2]
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            z_lin = torch.linspace(0, 1, z_dim)
            X, Y, Z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
            grid = torch.stack((X, Y, Z), dim=-1)  # shape: (x_dim, y_dim, 2)
            # pdb.set_trace()
            grid_aux = grid
            
            
        return (
            primary_slice,  # Primary input: shape [1, x, y, t, v] in test, [x,y,t,v] in training.
            primary_target,   # Primary target: full primary data.
            aux_slice,      # Auxiliary input: shape [1, x, y, t, v] in test, [3,x,y,t,v] in training.
            aux_target,       # Auxiliary target: full auxiliary data.
            grid,
            grid_aux,
        )