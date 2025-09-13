#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")


FILES = []
DATASET = 'data'
SLICE_AXIS = 1
OUT_DIR = '.'


def plot_last_timestep(h5_path: str) -> None:
    tag = os.path.splitext(os.path.basename(h5_path))[0]

    with h5py.File(h5_path, 'r') as f:
        last_frame = f[DATASET][0]

    cut = last_frame[:, last_frame.shape[SLICE_AXIS]//2, :]
    cut = np.rot90(cut, k=-1)

    ny, nx = cut.shape
    aspect_ratio = nx / ny

    base_height = 4
    fig_width = base_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(3, 6))

    im = ax.imshow(cut, vmin=0, vmax=np.nanmax(cut),
                   origin='lower', aspect='equal')

    ax.set_xticks([])
    ax.set_yticks([])                      # no axes
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.linspace(im.get_clim()[0], im.get_clim()[1],
                        min(4, 1 + int(im.get_clim()[0] != im.get_clim()[1])))
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f'{tag}_xz_init.png')
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'Saved → {out_png}')


for p in FILES:
    if os.path.isfile(p):
        plot_last_timestep(p)
    else:
        print(f'Warning: {p} not found, skipping.')
