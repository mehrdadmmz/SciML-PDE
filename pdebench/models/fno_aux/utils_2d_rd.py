from __future__ import annotations

import math as mt
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pdb


class FNODatasetMult(Dataset):
    def __init__(
        self,
        initial_step=10,
        saved_folder="../data_gen/data/",
        aux_saved_folder ="../data_gen/data/",
        if_test=False,
        if_downsample=False,
        train_subsample=[900, 900, 900],
        num_aux_samples=3,
        rollout_test=1,
        test_ratio=0.1,
    ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        self.primary_path = Path(saved_folder + "2D_diff-react_test_all" + ".h5").resolve()
        if if_downsample:
            self.aux_path = Path(saved_folder + "2D_diff-react_downsample_t50_96" + ".h5").resolve()
            # self.aux_path = Path(saved_folder + "2D_diff-react_decomp_downsample" + ".h5").resolve()
        else:
            self.aux_path = Path(saved_folder + "2D_diff-react_test_diff" + ".h5").resolve()
        
        self.train_subsample_primary = train_subsample[1]
        self.train_subsample_aux = train_subsample[2]
        self.if_test = if_test
        self.if_downsample = if_downsample
        self.initial_step = initial_step
        self.num_aux_samples = num_aux_samples
        self.rollout_test = rollout_test

        # Extract list of seeds
        with h5py.File(self.primary_path, "r") as h5_primary, \
             h5py.File(self.aux_path, "r") as h5_aux:
            primary_list = sorted(h5_primary.keys())
            aux_list = sorted(h5_aux.keys())

        test_idx = int(len(primary_list) * (1 - test_ratio))
        if if_test:
            self.primary_list = np.array(primary_list[test_idx:])
            self.aux_list = np.array(aux_list[test_idx:])
        else:
            self.primary_list = np.array(primary_list[:self.train_subsample_primary])
            self.aux_list = np.array(aux_list[:self.train_subsample_aux])

        # Build an index mapping for temporal slicing.
        # For training: each primary seed is pre-sliced into all possible windows.
        # For testing: use a fixed window (e.g. the first window).
        self.indices = []  # Each element will be a tuple: (primary_idx, t_start)
        if not self.if_test:
            # For each primary seed, determine T_total and store all valid start indices.
            for i, key in enumerate(self.primary_list):
                with h5py.File(self.primary_path, "r") as f:
                    data = np.array(f[key]["data"], dtype="f")
                T_total = data.shape[0]  # original time dimension
                for t_start in range(T_total - self.initial_step):
                    self.indices.append((i, t_start))
        else:
            # For testing, use a fixed window (e.g., t_start = 0) for each primary sample.
            for i, key in enumerate(self.primary_list):
                self.indices.append((i, 0))

        


    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, idx):
        # Open file and read data
        primary_idx, t_start = self.indices[idx]
        with h5py.File(self.primary_path, "r") as h5_primary, \
             h5py.File(self.aux_path, "r") as h5_aux:
            
             # ---------- Primary Data -----------
            seed_primary = h5_primary[self.primary_list[primary_idx]]
            # data dim = [t, x1, ..., xd, v]
            data_primary = np.array(seed_primary["data"], dtype="f")
            data_primary = torch.tensor(data_primary, dtype=torch.float)
            permute_idx = list(range(1, len(data_primary.shape) - 1))
            permute_idx.extend([0, -1]) # [x1, ..., xd, t, v]
            data_primary = data_primary.permute(permute_idx)
            primary_slice = data_primary[..., t_start:t_start+self.initial_step, :]
            primary_target = data_primary[..., (t_start+self.initial_step):(t_start+self.initial_step + self.rollout_test), :]


            # ------------ Auxiliary Data -----------
            if self.if_test:
                # In test mode, load one auxiliary sample and return as [1, x, y, t, v]
                seed_aux = h5_aux[self.aux_list[idx]]
                data_aux = np.array(seed_aux["data"], dtype="f")
                data_aux = torch.tensor(data_aux, dtype=torch.float)
                data_aux = data_aux.permute(permute_idx)  # [x, y, t, v]
                if self.if_downsample:
                    x_dim, y_dim, t_dim = data_primary.shape[0], data_primary.shape[1], data_primary.shape[2]
                    data_tf = data_aux.permute(3, 2, 0, 1).unsqueeze(0) # [1, 2, 51, 96, 96]
                    data_tf = F.interpolate(data_tf, size=(t_dim, x_dim, y_dim), mode="trilinear", align_corners=False)
                    data_aux = data_tf.permute(0, 3, 4, 2, 1).squeeze(0)
                aux_slice = data_aux[..., t_start:t_start+self.initial_step, :]
                aux_target = data_aux[..., (t_start+self.initial_step):(t_start+self.initial_step+ self.rollout_test), :]
                if aux_slice.dim() == 4:
                    aux_slice = aux_slice.unsqueeze(0)  # [1, x, y, t, v]
                    aux_target = aux_target.unsqueeze(0)
            else:
                # In training mode, group 3 auxiliary samples.
                aux_target_list = []
                aux_slice_list = []
                for i in range(self.num_aux_samples):
                    aux_idx = primary_idx * self.num_aux_samples  + i
                    seed_aux = h5_aux[self.aux_list[aux_idx]]
                    aux_data = np.array(seed_aux["data"], dtype="f")
                    aux_data = torch.tensor(aux_data, dtype=torch.float)
                    aux_data = aux_data.permute(permute_idx)  # [x, y, t, v]
                    if self.if_downsample:
                        x_dim, y_dim, t_dim = data_primary.shape[0], data_primary.shape[1], data_primary.shape[2]
                        data_tf = aux_data.permute(3, 2, 0, 1).unsqueeze(0) # [1, 2, 51, 96, 96]
                        data_tf = F.interpolate(data_tf, size=(t_dim, x_dim, y_dim), mode="trilinear", align_corners=False)
                        aux_data =data_tf.permute(0, 3, 4, 2, 1).squeeze(0)
                    aux_slice_list.append(aux_data[..., t_start:t_start+self.initial_step, :])
                    aux_target_list.append(aux_data[...,(t_start+self.initial_step):(t_start+self.initial_step+self.rollout_test), :])
                # Stack auxiliary samples to get [num_aux, x, y, t, v]
                aux_slice = torch.stack(aux_slice_list)
                aux_target = torch.stack(aux_target_list)

                
            # Extract spatial dimension of data
            dim = len(data_primary.shape) - 2
            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(seed_primary["grid"]["x"], dtype="f")
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
                grid_aux = grid
            elif dim == 2:
                x = np.array(seed_primary["grid"]["x"], dtype="f")
                y = np.array(seed_primary["grid"]["y"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing="ij")
                grid = torch.stack((X, Y), axis=-1)
                grid_aux = grid
            elif dim == 3:
                x = np.array(seed_primary["grid"]["x"], dtype="f")
                y = np.array(seed_primary["grid"]["y"], dtype="f")
                z = np.array(seed_primary["grid"]["z"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
                grid = torch.stack((X, Y, Z), axis=-1)

        return (
            primary_slice,  # Primary input: shape [1, x, y, t, v] in test, [x,y,t,v] in training.
            primary_target,   # Primary target: full primary data.
            aux_slice,      # Auxiliary input: shape [1, x, y, t, v] in test, [3,x,y,t,v] in training.
            aux_target,       # Auxiliary target: full auxiliary data.
            grid,
            grid_aux,
        )

