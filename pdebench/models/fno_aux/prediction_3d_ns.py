from __future__ import annotations

import pickle
from pathlib import Path
import os

import numpy as np
import torch
from fno_aux.fno_aux import FNO2d, FNO3d
from fno_aux.utils_3d_ns import FNODatasetMult
from metrics_aux import metrics
from torch import nn
from tqdm import tqdm
import wandb
import random
import pdb
import gc
import h5py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_training(
    if_training,
    if_downsample,
    continue_training,
    num_workers,
    rollout_test,
    modes,
    width,
    initial_step,
    t_train,
    num_channels,
    batch_size,
    epochs,
    train_subsample,
    learning_rate_share,
    learning_rate_fc2,
    num_aux_samples,
    auxiliary_weight,
    scheduler_step,
    scheduler_gamma,
    model_update,
    model_flmn,
    base_path,
    aux_path,
    plot,
    channel_plot,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max,
    training_type="single",
    scheduler="cosine",
):
    # print(
    #    f"Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}"
    # )

    ################################################################
    # load data
    ################################################################

    # filename
    model_name = model_flmn + "_aux_FNO"

    # print("FNODatasetMult")
    train_data = FNODatasetMult(
        saved_folder=base_path,
        aux_saved_folder = aux_path,
        if_test=False,
        train_subsample=train_subsample,
        num_aux_samples=num_aux_samples,
    )
    val_data = FNODatasetMult(
        saved_folder=base_path,
        aux_saved_folder = aux_path,
        if_test=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, persistent_workers=True
    )

    print("length of training loader:",len(train_loader), "length of test loader:", len(val_loader))
    print("Device:", device)

    _, _data, _, _, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    print("Spatial Dimension", dimensions - 3)

    if dimensions == 5:
        model = FNO2d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes,
            initial_step=initial_step,
        ).to(device)
    elif dimensions == 6:
        model = FNO3d(
            num_channels=num_channels,
            width=width,
            modes1=modes,
            modes2=modes,
            modes3=modes,
            initial_step=initial_step,
        ).to(device)


    # Set maximum time step of the data to train (shape[-2] corresponds to the time dimension)
    t_train = min(t_train, _data.shape[-2])

    model_path = model_name + ".pt"

    output_dir = "../data_gen/result_plot"

    if not if_training:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        with torch.no_grad():

            for i, (xx, yy, xx_aux, yy_aux, grid, grid_aux) in enumerate(tqdm(val_loader, desc="Inferring")):
                # send to device
                xx, yy = xx.to(device), yy.to(device)
                xx_aux, yy_aux = xx_aux.to(device), yy_aux.to(device)
                grid, grid_aux = grid.to(device), grid_aux.to(device)

                if dimensions == 5:
                    B, num_aux, X, Y, _, v = xx_aux.shape
                    xx_aux = xx_aux.reshape(B * num_aux, X, Y, -1, v)
                    yy_aux = yy_aux.reshape(B * num_aux, X, Y, -1, v)
                    grid_aux = grid_aux.unsqueeze(1).expand(-1, num_aux, -1, -1, -1)
                    grid_aux = grid_aux.reshape(B * num_aux, X, Y, -1) 
                elif dimensions == 6:
                    B, num_aux, X, Y, Z,  _, v = xx_aux.shape
                    xx_aux = xx_aux.reshape(B * num_aux, X, Y, Z, -1, v)
                    yy_aux = yy_aux.reshape(B * num_aux, X, Y, Z, -1, v)
                    grid_aux = grid_aux.unsqueeze(1).expand(-1, num_aux, -1, -1, -1, -1)
                    grid_aux = grid_aux.reshape(B * num_aux, X, Y, Z, -1) 

                # warm-start
                pred = yy[..., :initial_step, :]

                # rollout timesteps
                for t in range(initial_step, yy.shape[-2]):
                    im_primary, _ = model(xx, grid, xx_aux, grid_aux)
                    pred = torch.cat((pred, im_primary), dim=-2)
                    # shift xx / yy window
                    xx = torch.cat((xx[...,1:,:], yy[...,t:t+1,:]), dim=-2)

                # keep only the channel(s) you want
                pred = pred.squeeze(0).cpu()
                # e.g. if you want *all* channels:
                # pred_np = pred.permute(3,4,0,1,2).numpy()   # (T, C, X, Y, Z)
                # or if only channel 3:
                pred_np = pred[...,3].permute(3,0,1,2).numpy()

                # write out one file per sample
                out_path = os.path.join(output_dir, f"3D_NS_pred_trj_sample{i:03d}.h5")
                with h5py.File(out_path, "w") as f:
                    f.create_dataset("data", data=pred_np, compression="gzip")

                # cleanup
                del xx, yy, xx_aux, yy_aux, grid, grid_aux, pred, pred_np
                torch.cuda.empty_cache()
                gc.collect()

        print("Inference complete!")


    # if not if_training:
    #     checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     model.to(device)
    #     model.eval()
    #     with torch.no_grad():
    #         for xx, yy, xx_aux, yy_aux, grid, grid_aux in val_loader:
    #             xx = xx.to(device)  # noqa: PLW2901
    #             yy = yy.to(device)  # noqa: PLW2901
    #             xx_aux = xx_aux.to(device)
    #             yy_aux = yy_aux.to(device)
    #             grid = grid.to(device)  # noqa: PLW2901
    #             grid_aux = grid_aux.to(device)
    #             # pdb.set_trace()

    #             B, num_aux, X, Y, Z,  _, v = xx_aux.shape
    #             xx_aux = xx_aux.reshape(B * num_aux, X, Y, Z, -1, v)
    #             yy_aux = yy_aux.reshape(B * num_aux, X, Y, Z, -1, v)
    #             grid_aux = grid_aux.unsqueeze(1).expand(-1, num_aux, -1, -1, -1, -1)
    #             grid_aux = grid_aux.reshape(B * num_aux, X, Y, Z, -1) 


    #             pred = yy[..., :initial_step, :]

    #             for t in range(initial_step, yy.shape[-2]):
    #                 y = yy[..., t : t + 1, :]
    #                 im_primary, _ = model(xx, grid, xx_aux, grid_aux) 

    #                 pred = torch.cat((pred, im_primary), -2)
    #                 xx = torch.cat((xx[..., 1:, :],y), dim=-2)
                
    #             pred = pred.squeeze(0)
    #             pred = pred[...,3]
    #             pred = pred.permute(3, 0, 1, 2)
    #             pred = pred.cpu().numpy()
    #             with h5py.File(out_s, "w") as f:
    #                 f.create_dataset("data", data=pred, compression="gzip")
    #             print(f"[{os.getpid()}] Wrote {out_s}, shape {pred.shape}")