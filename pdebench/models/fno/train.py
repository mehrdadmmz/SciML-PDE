from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from fno.fno import FNO2d, FNO3d
from fno.utils_2d_ns_baseline_lie import FNODatasetMult
from metrics import metrics
from torch import nn
from tqdm import tqdm
import wandb
import random
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

seed =16
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
    learning_rate,
    scheduler_step,
    scheduler_gamma,
    model_update,
    FNO_model_flmn,
    plot,
    channel_plot,
    x_min,
    x_max,
    y_min,
    y_max,
    t_min,
    t_max,
    base_path="../data/",
    training_type="single",
    scheduler='cosine',
):
    # print(
    #    f"Epochs = {epochs}, learning rate = {learning_rate}, scheduler step = {scheduler_step}, scheduler gamma = {scheduler_gamma}"
    # )

    ################################################################
    # load data
    ###############################################################
    # filename
    model_name = FNO_model_flmn + "_FNO"

    # print("FNODatasetMult")
    train_data = FNODatasetMult(
        saved_folder=base_path,
        train_subsample=train_subsample,
        rollout_test=rollout_test,
    )
    val_data = FNODatasetMult(
        if_test=True,
        saved_folder=base_path,
        rollout_test=rollout_test,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    print("length of training loader:",len(train_loader), "length of test loader:", len(val_loader))
    print("Device:", device)

    ################################################################
    # training and evaluation
    ################################################################

    _, _data, _ = next(iter(val_loader))
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

    # Set maximum time step of the data to train
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

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total parameters = {total_params}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    # )
    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * (len(train_data) / batch_size))
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    
    # loss_fn = nn.MSELoss(reduction="mean")
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
        
    wandb.init(project="2D_NS_compare", config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "sample size" : train_subsample,
        "training type": training_type,
        "seed": seed,
    })


    # for ep in range(start_epoch, epochs):
    for ep in tqdm(range(start_epoch, epochs), desc="Training Progress"):
        model.train()
        # t1 = default_timer()
        train_l2_step = 0
        train_l2_full = 0
        for xx, yy, grid in train_loader:
            loss = 0

            # xx: input tensor (first few time steps) [b, x1, ..., xd, t_init, v]
            # yy: target tensor [b, x1, ..., xd, t, v]
            # grid: meshgrid [b, x1, ..., xd, dims]
            xx = xx.to(device)  # noqa: PLW2901
            yy = yy.to(device)  # noqa: PLW2901
            grid = grid.to(device)  # noqa: PLW2901

            # Initialize the prediction tensor
            pred = yy[..., :initial_step, :]

            if training_type in ["autoregressive"]:
                for t in range(initial_step, t_train):
                    y = yy[..., t : t + 1, :]
                    # Model 
                    im = model(xx, grid)
                    # Loss calculation
                    loss_primary = nrmse(im, y)
                    loss += loss_primary.mean()
                    pred = torch.cat((pred, im), -2)
                    xx = torch.cat((xx[..., 1:, :],  y), dim=-2)  # noqa: PLW2901

                train_l2_step += loss.item()
                _yy = yy[..., :t_train, :]  
                loss_full_primary = nrmse(pred, _yy)
                train_l2_full += loss_full_primary.mean().item()

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
                im = model(xx, grid)
                # Loss calculation
                loss_primary = nrmse(im, yy)
                loss += loss_primary.mean()

                train_l2_full += loss.item()

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
            val_l2_step = 0
            val_l2_full = 0
            with torch.no_grad():
                for xx, yy, grid in val_loader:
                    loss = 0
                    xx = xx.to(device)  # noqa: PLW2901
                    yy = yy.to(device)  # noqa: PLW2901
                    grid = grid.to(device)  # noqa: PLW2901

                    if training_type in ["autoregressive"]:
                        pred = yy[..., :initial_step, :]

                        for t in range(initial_step, t_train):
                            y = yy[..., t : t + 1, :]
                            # Model 
                            im = model(xx, grid)
                            # Loss calculation
                            loss_primary = nrmse(im, y)
                            loss += loss_primary.mean()
                            pred = torch.cat((pred, im), -2)
                            xx = torch.cat((xx[..., 1:, :],  y), dim=-2)  # noqa: PLW2901

                        val_l2_step += loss.item()
                        _pred = pred[..., initial_step:t_train, :]
                        _yy = yy[..., initial_step:t_train, :]
                        loss_full_primary = nrmse(_pred, _yy)
                        val_l2_full += loss_full_primary.mean().item()

                    if training_type in ["single"]:
                        im = model(xx, grid)
                        # Loss calculation
                        loss_primary = nrmse(im, yy)
                        loss += loss_primary.mean()
                        val_l2_full += loss.item()


                if val_l2_full < loss_val_min:
                    loss_val_min = val_l2_full
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
            "Train Primary Loss": train_l2_full/len(train_loader),
            "Validation Primary Loss": val_l2_full/len(val_loader),
            "BackboneLearning Rate": backbone_lr,
            "Gradient Norm": total_norm.item(),
            "Clipped Norm": clipped_norm.item()
        })

        # t2 = default_timer()
        scheduler.step()
        print(
           "epoch: {0}, loss: {1:.5f},  trainL2: {2:.5f}, testL2: {3:.5f}".format(
               ep, loss.item(), train_l2_full, val_l2_full
           )
        )
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_training()
    # print("Done.")
