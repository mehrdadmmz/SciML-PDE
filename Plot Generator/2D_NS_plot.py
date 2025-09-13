import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")


FILES = []
DATASET_VEL = "velocity"
# DATASET_FORCE = "force"
BATCH_IDX = 0
OUT_DIR = "."
CBAR_FSIZE = 20


def _save_field_png(field: np.ndarray, tag: str, comp: str) -> None:
    """Render a 2-D field (u, v, or force) to PNG with square pixels."""
    ny, nx = field.shape
    aspect = nx / ny
    base_h = 4
    fig, ax = plt.subplots(figsize=(base_h * aspect, base_h))

    im = ax.imshow(
        field,
        origin="lower",
        aspect="equal",
        vmin=np.nanmin(field),
        vmax=np.nanmax(field),
        cmap="turbo",
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.linspace(im.get_clim()[0], im.get_clim()[1],
                        min(4, 1 + int(im.get_clim()[0] != im.get_clim()[1])))
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=CBAR_FSIZE)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{tag}_{comp}_half.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved → {fname}")


def process_file(h5_path: str) -> None:
    tag = os.path.splitext(os.path.basename(h5_path))[0]
    with h5py.File(h5_path, "r") as f:
        vel = f[DATASET_VEL][BATCH_IDX]
        last = vel[500]
        u, v = last[:, :, 0], last[:, :, 1]
        _save_field_png(u.T, tag, "u")
        _save_field_png(v.T, tag, "v")

        # if DATASET_FORCE in f:
        #     force = f[DATASET_FORCE][BATCH_IDX]
        #     mag = np.sqrt(force[:, :, 0]**2 + force[:, :, 1]**2)
        #     _save_field_png(mag.T, tag, "force")


def main():
    for path in FILES:
        if os.path.isfile(path):
            process_file(path)
        else:
            print(f"Warning: {path} not found, skipping.")


if __name__ == "__main__":
    main()
