#!/usr/bin/env python3
# make_npy_from_h5.py  – robust converter for ns_aux velocity files
import h5py
import numpy as np
import glob
import os
from scipy.ndimage import zoom

SRC_DIR = "/local-scratch/pdebench/simulated/ns_aux"
OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)

h5_paths = sorted(glob.glob(f"{SRC_DIR}/*.h5"))
assert h5_paths, "No .h5 files found under ns_aux"


def to_mag64(arr):
    """
    Accept velocity arrays of rank 3-5 and return list of (64,64) frames.
    Layouts handled:
        (T, 256,256, 2)
        (case, T, 256,256, 2)
        (2, T, 256,256)
        (2, case, T, 256,256)
        (256,256,2)         etc.
    """
    a = np.array(arr)                    # make write-able copy
    # --- move channel axis to last --------------------------------
    if a.shape[-1] == 2:                 # already channel-last
        pass
    elif a.shape[0] == 2:                # channel-first
        a = np.moveaxis(a, 0, -1)
    else:
        raise ValueError(f"can't find 2-channel axis in shape {a.shape}")

    # now a has shape (..., T, 256,256, 2) or (T,256,256,2)
    if a.ndim == 5:          # (case, T, H, W, 2)
        case, T, H, W, _ = a.shape
        a = a.reshape(case * T, H, W, 2)
    elif a.ndim == 4:        # (T, H, W, 2)
        T, H, W, _ = a.shape
    elif a.ndim == 3:        # (H, W, 2)  -> single frame
        a = a[None]          # add time dim
        T, H, W, _ = a.shape
    else:
        raise ValueError(f"unexpected rank {a.ndim}")

    # magnitude
    mag = np.linalg.norm(a, axis=-1)     # (frames, H, W)

    # down-sample if needed
    if H != 64 or W != 64:
        scale = (1, 64 / H, 64 / W)
        mag = zoom(mag, scale, order=3)

    return list(mag)                     # list of (64,64) frames


# ------------------------------------------------------------------
frames = []
for p in h5_paths:
    with h5py.File(p, "r") as f:
        frames += to_mag64(f["velocity"][:])
    if len(frames) >= 50:
        break

assert len(frames) >= 50, "collected only %d frames" % len(frames)
sample = np.stack(frames[:50], axis=0)        # (50,64,64)
out_file = os.path.join(OUT_DIR, "ns_aux_mag64_T50.npy")
np.save(out_file, sample[..., None])          # trailing sample dim
print("✅ Saved", sample.shape, "→", out_file)
