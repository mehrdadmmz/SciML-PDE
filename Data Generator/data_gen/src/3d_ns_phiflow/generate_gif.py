import os
import imageio.v2 as imageio
import numpy as np
import shutil
import matplotlib.pyplot as plt
import h5py

sname = 'smoke'
# trj = np.load('v_trj.npy') # ('x', 'y', 'z', 'time', 'vector')
# T = trj.shape[3]
#f = h5py.File('/local-scratch1/SciML-PDE/pdebench/data_gen/3D_scalarflow_interp/s_trj_seed0.h5',"r") # ('time', 'x', 'y', 'z')
f = h5py.File('/local-scratch1/SciML-PDE/pdebench/data_gen/pred_trj_baseline_sample000.h5',"r") # ('time', 'x', 'y', 'z')
trj = f['data'][:]
T = trj.shape[0]
print(trj.shape)
y_slice = trj.shape[2]//2

print(np.isnan(trj).sum())

imgs = []
for i in range(T):
    fig, axs = plt.subplots(ncols=1, figsize=(11, 4))
    # volume = trj[:, 32, :, i, -1] # vtrj
    volume = trj[i, :, y_slice, :] # strj
    vmin = volume.min()
    vmax = volume.max()
    a = axs.imshow(volume, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(a, ax=axs)
    cbar.ax.tick_params(labelsize=14)
    # plt.savefig(f'./{sname}_{i}.png')
    plt.savefig(f'./{sname}.png')
# for i in range(T):
    # fig = imageio.imread(f'./{sname}_{i}.png')
    fig = imageio.imread(f'./{sname}.png')
    imgs.append(fig)
print(f'length of frames: {len(imgs)}')
imageio.mimsave(f'./{sname}_sim_pred_baseline.gif', imgs, duration=.1)

os.system(f'rm ./{sname}.png')
# for i in range(T):
#     os.system(f'rm ./{sname}_{i}.png')
