from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from fno_aux.fno_aux import FNO2d, FNO3d
from fno_aux.utils_2d_ns import FNODatasetMult
from metrics_aux import metrics
from torch import nn
from tqdm import tqdm
import wandb
import random
import pdb
import gc

# torch.manual_seed(0)
# np.random.seed(0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed=16
set_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nrmse(output, tar):
    spatial_dims = tuple(range(output.ndim))[1:4] # [b, h, w, t ,c]
    residuals = output - tar
    # Differentiate between log and accumulation losses
    tar_norm = (1e-7 + tar.pow(2).mean(spatial_dims, keepdim=True))
    raw_loss = ((residuals).pow(2).mean(spatial_dims, keepdim=True) / tar_norm)
    return raw_loss

def run_training(
    if_training,
    if_downsample,
    continue_training,
    rollout_test,
    num_workers,
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

    model_name = model_flmn + "_aux_FNO"

    train_data = FNODatasetMult(
        saved_folder=base_path,
        aux_saved_folder = aux_path,
        if_test=False,
        if_downsample=if_downsample,
        train_subsample=train_subsample,
        num_aux_samples=num_aux_samples,
        rollout_test=rollout_test,
    )
    val_data = FNODatasetMult(
        saved_folder=base_path,
        aux_saved_folder = aux_path,
        if_test=True,
        if_downsample=if_downsample,
        rollout_test=rollout_test,
    )


    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=False, persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False, persistent_workers=True
    )

    print("length of training loader:",len(train_loader), "length of test loader:", len(val_loader))
    print("Device:", device)

    ################################################################
    # training and evaluation
    ################################################################
    # retrieve the next element from the iterator
    _, _data, _, _, _, _ = next(iter(val_loader))
    dimensions = len(_data.shape)
    # print("Spatial Dimension", dimensions - 3)
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

    if not if_training:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        Lx, Ly, Lz = 1.0, 1.0, 1.0
        errs = metrics(
            val_loader,
            model,
            rollout_test,
            Lx,
            Ly,
            Lz,
            plot,
            channel_plot,
            model_name,
            x_min,
            x_max,
            y_min,
            y_max,
            t_min,
            t_max,
            initial_step=initial_step,
        )
        with Path(model_name + ".pickle").open("wb") as pb:
            pickle.dump(errs, pb)

        return

    optimizer = torch.optim.Adam([
        {'params': model.shared_layers.parameters(), 'lr': learning_rate_share, 'weight_decay': 1e-4},         # Shared layers
        {'params': model.fc2_primary.parameters(), 'lr': learning_rate_fc2, 'weight_decay': 1e-4},           # Primary task head
        {'params': model.fc2_auxiliary.parameters(), 'lr': learning_rate_fc2, 'weight_decay': 1e-4}          # Auxiliary task head
    ])


    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs * (len(train_data)/ batch_size))  
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    loss_val_min = np.inf

    start_epoch = 0

    

    # If desired, restore the network by loading the weights saved in the .pt
    # file
    if continue_training:
        # print("Restoring model (that is the network's weights) from file...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.train()

        # Load optimizer state dict
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint["epoch"]
        loss_val_min = checkpoint["loss"]

    
    #WandB set up
    wandb.init(project="2D_RD_compare", config={
        "seed": seed,
        "learning_rate_share": learning_rate_share,
        "learning_rate_fc2": learning_rate_fc2,
        "epochs": epochs,
        "batch_size": batch_size,
        "sample size" : train_subsample,
        "training type": training_type,
        "auxiliary_weight": auxiliary_weight,
        "if_downsample": if_downsample,
    })


    #for ep in range(start_epoch, epochs):
    for ep in tqdm(range(start_epoch, epochs), desc="Training Progress"):
        model.train() # set the model to training mode
        # t1 = default_timer()
        train_l2_full_primary = 0
        train_l2_full_auxiliary = 0
        # train_l2_full = 0
        for xx, yy, xx_aux, yy_aux, grid, grid_aux in train_loader:
            loss = 0
            train_primary_loss = 0
            train_auxiliary_loss = 0

            # xx, xx_aux: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # yy, yy_aux: target tensor [b, x1, ..., xd, t, v]
            # grid, grid_aux: meshgrid [b, x1, ..., xd, dims]
            xx = xx.to(device)  # noqa: PLW2901
            yy = yy.to(device)  # noqa: PLW2901
            xx_aux = xx_aux.to(device)
            yy_aux = yy_aux.to(device)
            grid = grid.to(device)  # noqa: PLW2901
            grid_aux = grid_aux.to(device)
            # pdb.set_trace()

            #  Reshape the input tensor to [B*num_aux, X, Y, T, V]
            B, num_aux, X, Y, _, v = xx_aux.shape
            xx_aux = xx_aux.reshape(B * num_aux, X, Y, -1, v)
            yy_aux = yy_aux.reshape(B * num_aux, X, Y, -1, v)
            grid_aux = grid_aux.unsqueeze(1).expand(-1, num_aux, -1, -1, -1)
            grid_aux = grid_aux.reshape(B * num_aux, X, Y, -1) 

            # Initialize the prediction tensor (0 to initial_step-1)
            # pred = yy[..., :initial_step, :]
            # pred_aux = yy_aux[..., :initial_step, :]


            if training_type in ["autoregressive"]:
                # Autoregressive loop
                for t in range(initial_step, t_train):
                    # Extract target at current time step (t-th time step)
                    y = yy[..., t : t + 1, :]
                    y_aux = yy_aux[..., t : t + 1, :]

                    # Run model for primary data and auxiliary data
                    im_primary, im_auxiliary= model(xx, grid, xx_aux, grid_aux)
                    
                    loss_primary = nrmse(im_primary, y)
                    loss_auxiliary = nrmse(im_auxiliary, y_aux)
                    train_primary_loss += loss_primary.mean()
                    train_auxiliary_loss += loss_auxiliary.mean()

                    # Concatenate the prediction at current time step into the prediction tensor
                    pred = torch.cat((pred, im_primary), -2)
                    pred_aux = torch.cat((pred_aux, im_auxiliary), -2)

                    # only use the ground truth data
                    xx = torch.cat((xx[..., 1:, :], y), dim=-2) # noqa: PLW2901
                    xx_aux = torch.cat((xx_aux[..., 1:, :], y_aux), dim=-2)
                    
                loss = train_primary_loss + auxiliary_weight * train_auxiliary_loss
                # accumulate the loss at each time step
                train_l2_step_primary += train_primary_loss.item()
                train_l2_step_auxiliary += train_auxiliary_loss.item()
                _yy = yy[..., :t_train, :]  # if t_train is not -1
                _yy_aux = yy_aux[..., :t_train, :]
                loss_full_primary = nrmse(pred, _yy)
                loss_full_auxiliary = nrmse(pred_aux, _yy_aux)
                train_l2_full_primary += loss_full_primary.mean().item()
                train_l2_full_auxiliary += loss_full_auxiliary.mean().item()
               
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
                clip_value = max(5, 0.1 * total_norm)  # Set clip value as 10% of norm, but at least 5
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                clipped_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
                optimizer.step()
                scheduler.step()
                backbone_lr = optimizer.param_groups[0]['lr']

            if training_type in ["single"]:
                im_primary, im_auxiliary= model(xx, grid, xx_aux, grid_aux)
                    
                loss_primary = nrmse(im_primary, yy)
                loss_auxiliary = nrmse(im_auxiliary, yy_aux)
                train_primary_loss += loss_primary.mean()
                train_auxiliary_loss += loss_auxiliary.mean()
                    
                loss = train_primary_loss + auxiliary_weight * train_auxiliary_loss
                # accumulate the loss at each time step000
                train_l2_full_primary += train_primary_loss.item()
                train_l2_full_auxiliary += train_auxiliary_loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
                clip_value = max(5, 0.1 * total_norm)  # Set clip value as 10% of norm, but at least 5
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                clipped_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters() if p.grad is not None]), 2)
                optimizer.step()
                scheduler.step()
                backbone_lr = optimizer.param_groups[0]['lr']



        if ep % model_update == 0:

            val_l2_full_primary = 0
            val_l2_full_auxiliary = 0

            with torch.no_grad():
                for xx, yy, xx_aux, yy_aux, grid, grid_aux in val_loader:
                    loss = 0
                    val_primary_loss = 0
                    val_auxiliary_loss = 0
                    xx = xx.to(device)  # noqa: PLW2901
                    yy = yy.to(device)  # noqa: PLW2901
                    xx_aux = xx_aux.to(device)
                    yy_aux = yy_aux.to(device)
                    grid = grid.to(device)  # noqa: PLW2901
                    grid_aux = grid_aux.to(device)

                    B, num_aux, X, Y, _, v = xx_aux.shape
                    xx_aux = xx_aux.reshape(B * num_aux, X, Y, -1, v)
                    yy_aux = yy_aux.reshape(B * num_aux, X, Y, -1, v)
                    grid_aux = grid_aux.unsqueeze(1).expand(-1, num_aux, -1, -1, -1)
                    grid_aux = grid_aux.reshape(B * num_aux, X, Y, -1) 

                    
                    if training_type in ["autoregressive"]:
                        pred = yy[..., :initial_step, :]
                        pred_aux = yy_aux[..., :initial_step, :]

                        for t in range(initial_step, yy.shape[-2]):
                            y = yy[..., t : t + 1, :]
                            y_aux = yy_aux[..., t : t + 1, :]
                            im_primary, im_auxiliary = model(xx, grid, xx_aux, grid_aux) 
                            loss_primary = nrmse(im_primary, y)
                            loss_auxiliary = nrmse(im_auxiliary, y_aux)
                            val_primary_loss += loss_primary.mean()
                            val_auxiliary_loss += loss_auxiliary.mean()

                            pred = torch.cat((pred, im_primary), -2)
                            pred_aux = torch.cat((pred_aux, im_auxiliary), -2)
                            xx = torch.cat((xx[..., 1:, :],y), dim=-2) # noqa: PLW2901
                            xx_aux = torch.cat((xx_aux[..., 1:, :], y_aux), dim=-2)

                        loss = val_primary_loss + auxiliary_weight * val_auxiliary_loss 
                        val_l2_step_primary += val_primary_loss.item()
                        val_l2_step_auxiliary += val_auxiliary_loss.item()
                        _pred = pred[..., initial_step:t_train, :]
                        _pred_aux = pred_aux[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]
                        _yy_aux = yy_aux[..., initial_step:t_train, :]
                        loss_full_primary = nrmse(_pred, _yy)
                        loss_full_auxiliary = nrmse(_pred_aux, _yy_aux)
                        val_l2_full_primary += loss_full_primary.mean().item()
                        val_l2_full_auxiliary += loss_full_auxiliary.mean().item()

                    if training_type in ["single"]:

                        im_primary, im_auxiliary = model(xx, grid, xx_aux, grid_aux) 
                        loss_primary = nrmse(im_primary, yy)
                        loss_auxiliary = nrmse(im_auxiliary, yy_aux)
                        val_primary_loss += loss_primary.mean()
                        val_auxiliary_loss += loss_auxiliary.mean()

                        loss = val_primary_loss + auxiliary_weight * val_auxiliary_loss 
                        val_l2_full_primary += val_primary_loss.item()
                        val_l2_full_auxiliary += val_auxiliary_loss.item()
                

                if val_l2_full_primary < loss_val_min:
                    loss_val_min = val_l2_full_primary
                    torch.save(
                        {
                            "epoch": ep,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": loss_val_min,
                        },
                        model_path,
                    )

        wandb.log({
            "Train Primary Loss": train_l2_full_primary/len(train_loader),
            "Train Auxiliary Loss": train_l2_full_auxiliary/len(train_loader),
            "Validation Primary Loss": val_l2_full_primary/len(val_loader),
            "Validation Auxiliary Loss": val_l2_full_auxiliary/len(val_loader),
            "BackboneLearning Rate": backbone_lr,
            "Gradient Norm": total_norm.item(),
            "Clipped Norm": clipped_norm.item()
        })

        # t2 = default_timer()
        scheduler.step()
        print(
           "epoch: {0}, loss: {1:.5f},  trainL2: {2:.5f}, trainL2_AUX: {3:.5f}, testL2: {4:.5f}, testL2_AUX: {5:.5f}".format(
               ep, loss.item(), train_l2_full_primary, train_l2_full_auxiliary, val_l2_full_primary, val_l2_full_auxiliary
           )
        )
        gc.collect()
        torch.cuda.empty_cache()



if __name__ == "__main__":
    nn.multiprocessing.set_start_method('spawn', force=True)
    run_training()
    # print("Done.")
