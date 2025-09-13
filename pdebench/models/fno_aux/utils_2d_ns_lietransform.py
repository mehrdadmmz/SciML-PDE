from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from fno.transformations import NSTransforms
import pdb

transforms_strength = [
    0.1,        # g₁: time shift ±10% of crop_t
    0.1,        # g₂: x-translation ±10% of x-domain
    0.1,        # g₃: y-translation ±10%
    0.05,       # g₄: scaling ±5%
    math.pi/18, # g₅: rotation ±10°
    0.2,        # g₆: x-Galilean boost ±20% of RMS(u)
    0.2,        # g₇: y-Galilean boost ±20% of RMS(v)
    0.05,       # g₈: x-quadratic boost ±5%
    0.05        # g₉: y-quadratic boost ±5%
]

class RandomCrop3d(torch.nn.Module):
    def __init__(self, crop_size):
        super().__init__()
        self.crop_size = crop_size

    def forward(self, tensor):
        C, T, H, W = tensor.size()
        t, h, w = self.crop_size

        if t > T or h > H or w > W:
            raise ValueError("Crop size must be smaller than input size")

        left = torch.randint(0, W - w + 1, size=(1,))
        top = torch.randint(0, H - h + 1, size=(1,))
        start = torch.randint(0, T - t + 1, size=(1,))

        right = left + w
        bottom = top + h
        end = start + t

        return tensor[..., start:end, top:bottom, left:right]

class LPSNavierStokes(object):
    def __init__(
              self,
              transforms_strength: Optional[List[float]] = transforms_strength,
              steps: Optional[int] = 2,
              order: Optional[int] = 2,
              crop_size: Optional[Tuple[int]] = (11, 256, 256),
              ) -> None:

        self.transforms_strength = transforms_strength
        self.crop = RandomCrop3d(crop_size)
        self.steps = steps
        self.order = order

    def __call__(
            self,
            sample: torch.Tensor,
            ) -> torch.Tensor:

        x, y, t, vx, vy = sample

        lie_transform = NSTransforms()

        vals = []
        vals.append(np.random.uniform(0, self.transforms_strength[0]))
        for strength in self.transforms_strength[1:]:
            vals.append(np.random.uniform(-strength, strength))

        if self.steps == 0:
            t_2, x_2, y_2, vx_2, vy_2 = t,x,y,vx,vy
        else:
            t_2, x_2, y_2, vx_2, vy_2 = lie_transform.apply(
                torch.tensor(vals),
                t,
                x,
                y,
                vx,
                vy,
                order=self.order,
                steps=self.steps,
                )
        image = torch.stack((x_2, y_2, t_2, vx_2, vy_2)).to(torch.float32)
        image = self.crop(image)

        return image
    

class FNODatasetMult(Dataset):
    def __init__(
        self,
        initial_step=10,
        saved_folder="../data_gen/ns_256/",
        aux_saved_folder="../data_gen/basic_eq",
        if_downsample=False,
        if_test=False,
        train_subsample=[900, 900, 900],
        num_aux_samples=24,
        rollout_test=1,
        # --- Lie‐symmetry args ---
        transforms_strength: list[float] = transforms_strength,
        steps: int = 2,
        order: int = 2,
    ):
        self.folder    = Path(saved_folder).resolve()
        self.aux_folder= Path(aux_saved_folder).resolve()
        self.initial_step   = initial_step
        self.if_downsample  = if_downsample
        self.if_test        = if_test
        self.num_aux_samples= num_aux_samples
        self.rollout_test   = rollout_test

        # --- instantiate your Lie augmentor ---
        self.lie_aug = LPSNavierStokes(
            transforms_strength=transforms_strength,
            steps=steps,
            order=order,
        )

       
        if if_test:
            self.primary_files = sorted([
                self.folder / f"ns_incom_inhom_2d_256-{i}.h5"
                for i in range(250, 275)
            ])
            self.aux_files = sorted([
                self.aux_folder / f"ns_aux_2d_256-{i}.h5"
                for i in range(250, 275)
            ])
        else:
            # For auxiliary files:
            self.aux_files = sorted([
                self.aux_folder / f"ns_aux_2d_256-{i}.h5"
                for i in range(train_subsample[2])
            ])
            # For primary files:
            if isinstance(train_subsample[1], float) and train_subsample[1] < 1:
                # When using a fraction, load the first file
                self.primary_files = sorted([self.folder / "ns_incom_inhom_2d_256-0.h5"])
            else:
                self.primary_files = sorted([
                    self.folder / f"ns_incom_inhom_2d_256-{i}.h5"
                    for i in range(train_subsample[1])
                ])
            # Group aux paths per primary
            num_primary = len(self.primary_files)
            self.aux_groups = [
                self.aux_files[i * num_aux_samples : (i + 1) * num_aux_samples]
                for i in range(num_primary)
            ]
        
        # Build index list: (file_idx, traj_idx, t_start)
        self.indices = []
        for file_idx, pth in enumerate(self.primary_files):
            # read shapes
            with h5py.File(pth, 'r') as f:
                B, T = f['velocity'].shape[:2]
            # determine number of trajectories
            if (not if_test) and isinstance(train_subsample[1], float) and train_subsample[1] < 1:
                num_traj = int(B * train_subsample[1])
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
        primary_full = torch.tensor(primary_np, dtype=torch.float)

        permute_idx = list(range(1, len(primary_np.shape) - 1))
        permute_idx.extend([0, -1]) # [x1, ..., xd, t, v]
        # primary_full = primary_np.permute(permute_idx)

        if not self.if_test:

            # build x,y,t coordinate tensors on [T, Y, X]
            x_dim, y_dim = primary_full.shape[1], primary_full.shape[2]
            T_steps = self.initial_step + self.rollout_test
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            t_lin = torch.linspace(0, 1, T_steps)

            X = torch.Tensor(np.tile(np.tile(x_lin.numpy(), (y_dim, 1)), (T_steps, 1, 1)))
            Y = torch.Tensor(np.tile(np.tile(y_lin.numpy(), (x_dim, 1)).T, (T_steps, 1, 1)))
            Tt = torch.Tensor(np.tile(t_lin.numpy(), (x_dim, y_dim, 1)).T)

            # extract u,v channels
            u = primary_full[..., 0]    # [T, X, Y]
            v = primary_full[..., 1]    # [T, X, Y]

            # apply Lie‐Trotter flow
            x2, y2, t2, u2, v2 = self.lie_aug((X, Y, Tt, u, v))

            # rebuild a 3‐channel stack:
            par_chan = primary_full[..., 2]  # [T, X, Y]
            primary_aug = torch.stack([u2, v2, par_chan], dim=-1)
            primary_full = primary_aug.permute(1,2,0,3)  # [X, Y, T, C]

        else:
            primary_full   = primary_full.permute(1,2,0,3) 

        primary_slice = primary_full[..., :self.initial_step, :]
        primary_target = primary_full[..., self.initial_step:(self.initial_step+self.rollout_test), :]

        # Load auxiliary on-demand
        if self.if_test:
            aux_pth = self.aux_files[file_idx]
            with h5py.File(aux_pth, 'r') as f:
                vel = f['velocity'][b, t0 : (t0 + self.initial_step + 1), ...]
                par = f['particles'][b, t0 : (t0 + self.initial_step + 1), ...]
            aux_np = np.concatenate([vel, par], axis=-1)
            aux_np = torch.tensor(aux_np, dtype=torch.float)
            aux_full = aux_np.permute(permute_idx) # shape: [X_aux, Y_aux, T+1, V]
            if self.if_downsample:
                x_dim, y_dim = primary_full.shape[0], primary_full.shape[1]
                aux_tf = aux_full.permute(2, 3, 0, 1)  # permute to [T+1, V, X_aux, Y_aux] so we can interpolate H,W
                aux_tf = F.interpolate(aux_tf, size=(x_dim, y_dim), mode="bilinear", align_corners=False) #[t, v, x_dim, y_dim]
                aux_full = aux_tf.permute(2, 3, 0, 1)
            aux_slice = aux_full[..., :self.initial_step, :].unsqueeze(0)
            aux_target = aux_full[..., self.initial_step:(self.initial_step+self.rollout_test), :].unsqueeze(0)
        else:
            aux_slice_list, aux_target_list = [], []
            for aux_pth in self.aux_groups[file_idx]:
                with h5py.File(aux_pth, 'r') as f:
                    vel = f['velocity'][b, t0 : (t0 + self.initial_step + 1), ...]
                    par = f['particles'][b, t0 : (t0 + self.initial_step + 1), ...]
                aux_np = np.concatenate([vel, par], axis=-1)
                aux_full = torch.tensor(aux_np, dtype=torch.float)
                aux_full = aux_full.permute(permute_idx)

                # x_dim, y_dim = aux_full.shape[1], aux_full.shape[2]
                # T_steps = self.initial_step + self.rollout_test
                # x_lin = torch.linspace(0, 1, x_dim)
                # y_lin = torch.linspace(0, 1, y_dim)
                # t_lin = torch.linspace(0, 1, T_steps)

                # X = torch.Tensor(np.tile(np.tile(x_lin.numpy(), (y_dim, 1)), (T_steps, 1, 1)))
                # Y = torch.Tensor(np.tile(np.tile(y_lin.numpy(), (x_dim, 1)).T, (T_steps, 1, 1)))
                # Tt = torch.Tensor(np.tile(t_lin.numpy(), (x_dim, y_dim, 1)).T)

                # # extract u,v channels
                # u = aux_full[..., 0]    # [T, X, Y]
                # v = aux_full[..., 1]    # [T, X, Y]

                # # apply Lie‐Trotter flow
                # x2, y2, t2, u2, v2 = self.lie_aug((X, Y, Tt, u, v))

                # # rebuild a 3‐channel stack:
                # par_chan = aux_full[..., 2]  # [T, X, Y]
                # aux_aug = torch.stack([u2, v2, par_chan], dim=-1)
                # aux_full = aux_aug.permute(1,2,0,3)  # [X, Y, T, C]

                if self.if_downsample:
                    x_dim, y_dim = primary_full.shape[0], primary_full.shape[1]
                    aux_tf = aux_full.permute(2, 3, 0, 1)  # permute to [T+1, V, X_aux, Y_aux] so we can interpolate H,W
                    aux_tf = F.interpolate(aux_tf, size=(x_dim, y_dim), mode="bilinear", align_corners=False) #[t, v, x_dim, y_dim]
                    aux_full = aux_tf.permute(2, 3, 0, 1)
                    
                aux_slice_list.append(aux_full[..., :self.initial_step, :])
                aux_target_list.append(aux_full[..., self.initial_step:(self.initial_step+self.rollout_test), :])
            aux_slice = torch.stack(aux_slice_list, dim=0)
            aux_target = torch.stack(aux_target_list, dim=0)


        # Extract spatial dimension of data
        dim = len(primary_full.shape) - 2

        if dim == 2:
            x_dim, y_dim = primary_full.shape[0], primary_full.shape[1]
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            X, Y = torch.meshgrid(x_lin, y_lin, indexing="ij")
            # pdb.set_trace()
            grid = torch.stack((X, Y), dim=-1)  # shape: (x_dim, y_dim, 2)
            grid_aux = grid
        elif dim == 3:
            x_dim, y_dim, z_dim = primary_full.shape[0], primary_full.shape[1], primary_full.shape[2]
            x_lin = torch.linspace(0, 1, x_dim)
            y_lin = torch.linspace(0, 1, y_dim)
            z_lin = torch.linspace(0, 1, z_dim)
            X, Y, Z = torch.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
            grid = torch.stack((X, Y, Z), dim=-1)  # shape: (x_dim, y_dim, 2)
            grid_aux = grid
            
            
        return (
            primary_slice,  # Primary input: shape [1, x, y, t, v] in test, [x,y,t,v] in training.
            primary_target,   # Primary target: full primary data.
            aux_slice,      # Auxiliary input: shape [1, x, y, t, v] in test, [3,x,y,t,v] in training.
            aux_target,       # Auxiliary target: full auxiliary data.
            grid,
            grid_aux,
        )