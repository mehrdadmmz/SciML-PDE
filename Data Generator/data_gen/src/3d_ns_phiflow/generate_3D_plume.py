import argparse
from phi.jax.flow import *
from tqdm import trange
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
from pdb import set_trace as bp
import jax.random as random
import h5py



def main(seed, path, res_x, res_y, res_z, DT, viscosity=0.1):

    # key = random.PRNGKey(seed)
    # Setting the random seed for simulation. This is a global seed for Phiflow.
    math.seed(seed)
    
    BND = extrapolation.ZERO
    bound = Box(x=(0, 1),y=(0, 1),z=(0, 1))
    
    smoke0 = CenteredGrid(0, extrapolation.BOUNDARY, x=res_x, y=res_y, z=res_z, bounds=bound) # sampled at cell centers
    
    v0 = StaggeredGrid(0, BND, x=res_x, y=res_y, z=res_z, bounds=bound) # sampled in staggered form at face centers
    
    sphere = Sphere(
        x=(bound._lower._native[0]+bound._upper._native[0])/2, y=(bound._lower._native[1]+bound._upper._native[1])/2, z=bound._lower._native[-1],
        radius=0.1*(bound._upper._native[0]-bound._lower._native[0]))
    INFLOW = 0.1*(bound._upper._native[0]-bound._lower._native[0]) * resample(sphere, to=smoke0, soft=True)
    
    
    
    
    @jit_compile
    def step(velocity, smoke, pressure, dt, diff=viscosity):
      smoke = advect.mac_cormack(smoke, velocity, dt=dt) + INFLOW
      velocity = advect.semi_lagrangian(velocity, velocity, dt=dt)
      velocity = diffuse.explicit(velocity, diffusivity=diff, dt=dt)
      import random as pyrandom
      f_x = pyrandom.random() * 0.0002 - 0.0001
      f_y = pyrandom.random() * 0.0002 - 0.0001
      buoyancy_force = (smoke * (f_x, f_y, 0.0005)).at(velocity) # shift
      velocity += buoyancy_force
      velocity = fluid.apply_boundary_conditions(velocity, ())
      velocity, pressure = fluid.make_incompressible(velocity, (), solve=Solve(method='CG', rel_tol=1e-3, x0=pressure))
      return velocity, smoke, pressure
    
    
    
    v_trj, s_trj, p_trj = iterate(step, batch(time=150), v0, smoke0, None, dt=DT, range=trange, substeps=10)
   # print(v_trj.shape, s_trj.shape)

    v_data = F.interpolate(torch.Tensor(v_trj.staggered_tensor().numpy(v_trj.shape.names)).cuda().permute(3, 4, 0, 1, 2), size=(50, 50, 89), mode='trilinear', align_corners=True).permute(2, 3, 4, 0, 1).cpu().numpy()
    s_data = F.interpolate(torch.Tensor(s_trj.values.numpy(s_trj.shape.names)).cuda().unsqueeze(1), size=(50, 50, 89), mode='trilinear', align_corners=True).squeeze(1).cpu().numpy()
    #print(v_data.shape, s_data.shape)
    v_data = v_data[..., 1:, :]
    s_data = s_data[1:, ...]
    #print(v_data.shape, s_data.shape)
    X, Y, Z, T, C = v_data.shape
    v_data =  F.interpolate(torch.Tensor(v_data).cuda().permute(0, 1, 2, 4, 3).reshape(-1, C, T), size= 150, mode='linear', align_corners=True).reshape(X, Y, Z, C, 150).permute(0, 1, 2, 4, 3).cpu().numpy() 
    T, X, Y, Z = s_data.shape
    s_data =  F.interpolate(torch.Tensor(s_data).cuda().permute(1, 2, 3, 0).reshape(-1, T).unsqueeze(1), size= 150, mode='linear', align_corners=True).squeeze(1).reshape(X, Y, Z, 150).permute(3, 0, 1, 2).cpu().numpy()
    #print(v_data.shape, s_data.shape)



    
    with h5py.File(f"{path}/v_trj_seed{seed}.h5", "w") as h5f:
        h5f.create_dataset("data", data=v_data, compression="gzip")

    with h5py.File(f"{path}/s_trj_seed{seed}.h5", "w") as h5f:
        h5f.create_dataset("data", data=s_data, compression="gzip")
    # np.savez_compressed(f"{path}/v_trj_seed{seed}.npy", data=F.interpolate(torch.Tensor(v_trj.staggered_tensor().numpy(v_trj.shape.names)).cuda().permute(3, 4, 0, 1, 2), size=(50, 50, 89), mode='trilinear', align_corners=True).permute(2, 3, 4, 0, 1).cpu().numpy())
    # np.savez_compressed(f"{path}/s_trj_seed{seed}.npy", data=F.interpolate(torch.Tensor(s_trj.values.numpy(s_trj.shape.names)).cuda().unsqueeze(1), size=(50, 50, 89), mode='trilinear', align_corners=True).squeeze(1).cpu().numpy())
    # np.save(f"{path}/v_trj_seed{seed}.npy", v_trj.staggered_tensor().numpy(v_trj.shape.names))
    # np.save(f"{path}/s_trj_seed{seed}.npy", s_trj.values.numpy(s_trj.shape.names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("3D NS")
    parser.add_argument('--path', type=str, default='/local-scratch1/3D_NS', help='Path to save simulations.')
    parser.add_argument('--seed', type=int, default=0, help='Seed.')
    parser.add_argument('--res-x', type=int, default=50, help='Resolution of x.')
    parser.add_argument('--res-y', type=int, default=50, help='Resolution of y.')
    parser.add_argument('--res-z', type=int, default=89, help='Resolution of z.')
    parser.add_argument('--dt', type=float, default=0.0002, help='dt.')
    parser.add_argument('--viscosity', type=float, default=0.001, help='viscosity.')
    args = parser.parse_args()
    print(args)
    main(args.seed, args.path, args.res_x, args.res_y, args.res_z, args.dt, args.viscosity)
