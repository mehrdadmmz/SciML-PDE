import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")

FILES = []
SAMPLE_IDX = 900
CHANNELS = {"U": 0, "V": 1}
OUT_DIR = "."
CBAR_FSIZE = 20
# --------------------------------------------------------------------------


def _save_png(field: np.ndarray, tag: str, label: str) -> None:
    """Render and save one 2-D field with clean styling."""
    ny, nx = field.shape
    aspect = nx / ny
    base_h = 4
    fig, ax = plt.subplots(figsize=(base_h * aspect, base_h))

    im = ax.imshow(
        field,
        origin="lower",
        aspect="equal",
        cmap="turbo",
        vmin=np.nanmin(field),
        vmax=np.nanmax(field),
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.linspace(im.get_clim()[0], im.get_clim()[1],
                        min(4, 1 + int(im.get_clim()[0] != im.get_clim()[1])))
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=CBAR_FSIZE)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"{tag}_{label}_half.png")
    plt.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved → {fname}")


def process_rd_file(h5_path: str) -> None:
    tag = os.path.splitext(os.path.basename(h5_path))[0]

    with h5py.File(h5_path, "r") as f:
        groups = sorted(k for k in f.keys() if k.isdigit())
        if not groups:
            print(f"⚠  No numeric groups in {h5_path}")
            return
        grp = groups[SAMPLE_IDX if SAMPLE_IDX < len(groups) else -1]
        data = f[f"{grp}/data"][:]

    last = data[51]
    for lbl, ch in CHANNELS.items():
        if ch < last.shape[2]:
            _save_png(last[:, :, ch].T, tag, lbl)


def main():
    for p in FILES:
        if os.path.isfile(p):
            process_rd_file(p)
        else:
            print(f"Warning: {p} not found, skipping.")


if __name__ == "__main__":
    main()
