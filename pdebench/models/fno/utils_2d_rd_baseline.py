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

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        self.file_path = Path(saved_folder + "2D_diff-react_test_all" + ".h5").resolve()
        self.train_subsample = train_subsample[0]
        self.if_test = if_test
        self.rollout_test = rollout_test

        # Extract list of seeds
        with h5py.File(self.file_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:self.train_subsample])

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.indices = []  # Each element will be a tuple: (primary_idx, t_start)
        if not self.if_test:
            # For each primary seed, determine T_total and store all valid start indices.
            for i, key in enumerate(self.data_list):
                with h5py.File(self.file_path, "r") as f:
                    data = np.array(f[key]["data"], dtype="f")
                T_total = data.shape[0]  # original time dimension
                for t_start in range(T_total - self.initial_step):
                    self.indices.append((i, t_start))
        else:
            # For testing, use a fixed window (e.g., t_start = 0) for each primary sample.
            for i, key in enumerate(self.data_list):
                self.indices.append((i, 0))



    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx, t_start = self.indices[idx]
        # Open file and read data
        with h5py.File(self.file_path, "r") as h5_file:
            seed_group = h5_file[self.data_list[data_idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype="f")
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape) - 1))
            permute_idx.extend([0, -1])
            data = data.permute(permute_idx)
            primary_slice = data[..., t_start:t_start+self.initial_step, :]
            primary_target = data[..., (t_start+self.initial_step):(t_start+self.initial_step+self.rollout_test), :]

            # Extract spatial dimension of data
            dim = len(data.shape) - 2

            # x, y and z are 1-D arrays
            # Convert the spatial coordinates to meshgrid
            if dim == 1:
                grid = np.array(seed_group["grid"]["x"], dtype="f")
                grid = torch.tensor(grid, dtype=torch.float).unsqueeze(-1)
            elif dim == 2:
                x = np.array(seed_group["grid"]["x"], dtype="f")
                y = np.array(seed_group["grid"]["y"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y, indexing="ij")
                grid = torch.stack((X, Y), axis=-1)
            elif dim == 3:
                x = np.array(seed_group["grid"]["x"], dtype="f")
                y = np.array(seed_group["grid"]["y"], dtype="f")
                z = np.array(seed_group["grid"]["z"], dtype="f")
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                z = torch.tensor(z, dtype=torch.float)
                X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
                grid = torch.stack((X, Y, Z), axis=-1)

        return primary_slice, primary_target, grid
