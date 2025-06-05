import os
import subprocess


gpu = 2
# seeds = list(range(150))


try:
    for seed in seeds:
        # Start the subprocess and wait for it to complete
        subprocess.run(
            f"XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES={gpu} python generate_3D_plume.py --path /local-scratch1/SciML-PDE/pdebench/data_gen/3D_ns_ood_v001 --seed {seed}",
            shell=True,
            check=True
        )
except KeyboardInterrupt:
    print("Process interrupted by user. Exiting.")
    # Optionally, perform any cleanup here
    exit(1)
