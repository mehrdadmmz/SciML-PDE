#!/usr/bin/env python3
# -----------------------------------------------------------
#  Evaluate a Hyena-Neural-Operator checkpoint on 2-D
#  reaction–diffusion and report five metrics:
#    • average relative L2 over the rollout
#    • accumulated MSE per frame
#    • relative L2 on the final frame
#    • nRMSE over the rollout
#    • nRMSE on the final frame
# -----------------------------------------------------------

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, repeat
from tqdm import tqdm

from loss_fn import rel_loss
from nn_module.encoder_module import SpatialTemporalEncoder2D
from nn_module.decoder_module import PointWiseDecoder2D        # ← matches ckpt
from nn_module.hyena_module import hyena1d


# ------------------------------------------------------------------ #
#  Helper functions
# ------------------------------------------------------------------ #
def build_model(opt):
    # Encoder: 12-→96-→192
    encoder = SpatialTemporalEncoder2D(
        input_channels=opt.in_channels,        # 12
        in_emb_dim=opt.encoder_emb_dim,    # 96
        out_seq_emb_dim=opt.out_seq_emb_dim,    # 192
        heads=opt.encoder_heads,      # 4
        depth=opt.encoder_depth       # 2
    )

    # Decoder: hybrid Fourier + Hyena block, width 384, 1 output step
    decoder = PointWiseDecoder2D(
        latent_channels=opt.decoder_emb_dim,    # 384
        out_channels=opt.out_channels,       # 1
        out_steps=opt.out_step,           # 1  (matches ckpt)
        propagator_depth=1,                      # ckpt has propagator.0 only
        scale=8
    )

    hyena = hyena1d(opt.out_seq_emb_dim)            # 192-wide bottleneck
    return encoder, decoder, hyena


def nrmse(pred, target, eps=1e-12):
    """normalised RMSE (per-batch)"""
    rmse = torch.sqrt(((pred - target)**2).mean(dim=-1) + eps)
    denom = (target.max(dim=-1).values -
             target.min(dim=-1).values).clamp(min=eps)
    return (rmse / denom).mean()


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True, type=str)
    parser.add_argument('--dataset_path',  required=True, type=str)

    # ---------- data / eval settings ----------
    parser.add_argument('--in_seq_len',    type=int, default=10)
    parser.add_argument('--out_seq_len',   type=int, default=40)
    parser.add_argument('--out_step',      type=int, default=1)   # critical
    parser.add_argument('--test_seq_num',  type=int, default=100)
    parser.add_argument('--batch_size',    type=int, default=16)
    parser.add_argument('--device',        type=str, default='cuda:0')

    # ---------- model h-params (override via CLI) ----------
    parser.add_argument('--in_channels',        type=int, default=12)
    parser.add_argument('--out_channels',       type=int, default=1)
    parser.add_argument('--encoder_emb_dim',    type=int, default=96)
    parser.add_argument('--decoder_emb_dim',    type=int, default=384)
    parser.add_argument('--out_seq_emb_dim',    type=int, default=192)
    parser.add_argument('--encoder_depth',      type=int, default=2)
    parser.add_argument('--encoder_heads',      type=int, default=4)
    opt = parser.parse_args()

    dev = torch.device(opt.device if torch.cuda.is_available() else 'cpu')

    # ---------- build & load networks ----------
    encoder, decoder, hyena = build_model(opt)
    ckpt = torch.load(opt.checkpoint, map_location=dev)
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder'])
    hyena.load_state_dict(ckpt['hyena_bottleneck'])

    encoder.to(dev).eval()
    decoder.to(dev).eval()
    hyena.to(dev).eval()

    # ---------- load dataset ----------
    data = np.load(opt.dataset_path)                # [T=50, 64, 64, N]
    ntest = opt.test_seq_num
    x_test = data[:opt.in_seq_len, ..., -ntest:]
    y_test = data[opt.in_seq_len:opt.in_seq_len+opt.out_seq_len, ..., -ntest:]

    x_test = rearrange(torch.tensor(x_test, dtype=torch.float32),
                       't h w n -> n t (h w)')
    y_test = rearrange(torch.tensor(y_test, dtype=torch.float32),
                       't h w n -> n t (h w)')

    loader = DataLoader(TensorDataset(x_test, y_test),
                        batch_size=opt.batch_size, shuffle=False)

    # ---------- normalise ----------
    x_mean, x_std = x_test.mean(), x_test.std()
    y_mean, y_std = y_test.mean(), y_test.std()
    x_test = (x_test - x_mean) / x_std

    # ---------- 64×64 coordinate grid ----------
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 64),
                                np.linspace(0, 1, 64)), axis=0)  # [2,64,64]
    grid = rearrange(torch.from_numpy(grid), 'c h w -> (h w) c')[None].float()

    # ---------- evaluation loop ----------
    avg_l2, acc_mse, last_l2 = [], [], []
    nrmse_all, nrmse_last = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, ncols=90, desc='Evaluating'):
            x, y = x.to(dev), y.to(dev)
            pos = repeat(grid.to(dev), '() n c -> b n c', b=x.size(0))

            z_in = torch.cat((rearrange(x, 'b t n -> b n t'), pos), dim=-1)
            z = hyena(encoder(z_in, pos))

            pred = decoder.rollout(z, pos, opt.out_seq_len, pos)
            pred = pred * y_std + y_mean                      # de-norm

            avg_l2.append(rel_loss(pred, y, p=2).item())
            acc_mse.append(F.mse_loss(pred, y, reduction='sum')
                           .item() / (y.size(0) * y.size(-1)))
            last_l2.append(rel_loss(pred[:, -1:], y[:, -1:], p=2).item())

            nrmse_all.append(nrmse(pred, y).item())
            nrmse_last.append(nrmse(pred[:, -1:], y[:, -1:]).item())

    # ---------- print results ----------
    print('\n----------------  RESULTS  ----------------')
    print(f'Avg relative L2 over seq  : {np.mean(avg_l2):.4e}')
    print(f'Accumulated MSE per frame : {np.mean(acc_mse):.4e}')
    print(f'Relative L2 @ final step  : {np.mean(last_l2):.4e}')
    print(f'nRMSE   over full rollout : {np.mean(nrmse_all):.4e}')
    print(f'nRMSE   on final frame    : {np.mean(nrmse_last):.4e}')


if __name__ == '__main__':
    main()
