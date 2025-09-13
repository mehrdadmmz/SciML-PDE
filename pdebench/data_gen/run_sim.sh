#!/bin/bash

# Set thread limits
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Define configuration file (can be different for each simulator)
cores="0-55"  # Specify CPU cores to use

# Output directories for .h5 files and logs
data_dir="data/simulated_data"
log_dir="data/logs"

# Create directories if they don't exist
mkdir -p $data_dir
mkdir -p $log_dir



for i in {1..5}; do
    echo "Trial $i"
    # Output file for logs
    log_file="$log_dir/all_trial${i}.txt"

    # Run the Hydra-based script with the specific configuration and output path
    taskset -c $cores /usr/bin/time -v python gen_diff_react.py \
        hydra.run.dir=$data_dir \
        output_path="$data_dir/all_trial${i}.h5" \
        >> "$log_file" 2>&1

    echo "Trial $i completed. Output saved to $data_dir/$all_trial${i}.h5"
done

