# Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge

This repository contains the official implementation of our **NeurIPS 2025** paper:

> **Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge**  
> [Paper Link — arXiv/NeurIPS to be uploaded]()

Our method enhances neural operator training by explicitly incorporating **fundamental physics knowledge** from decomposed PDE forms. By jointly training on both the original PDEs and their simplified basic forms (e.g., pure diffusion, convection), we achieve:

- **Higher data efficiency** (fewer simulations needed)
- **Improved long-term rollout consistency**
- **Stronger out-of-distribution (OOD) generalization**
- **Better synthetic-to-real transfer** (e.g., ScalarFlow smoke plumes)

---

## 📦 Installation

### Using pip

Locally:

```bash
pip install --upgrade pip wheel
pip install .
```

From PyPI:

```bash
pip install pdebench
```

To include dependencies for data generation:

```bash
pip install "pdebench[datagen310]"
pip install ".[datagen310]" # locally
```

or

```bash
pip install "pdebench[datagen39]"
pip install ".[datagen39]" # locally
```

### Using conda

If you like you can also install dependencies using anaconda, we suggest to use
[mambaforge](https://github.com/conda-forge/miniforge#mambaforge) as a
distribution.

Otherwise you may have to **enable the conda-forge** channel for the following commands.

Start with a fresh environment:

```bash
conda create -n myenv python=3.9
conda activate myenv
```

Install dependencies for model training:

```bash
conda install deepxde hydra-core h5py -c conda-forge
```

Install PyTorch (CUDA or CPU):

- CUDA 11.7 (Linux):

[see previous-versions/#linux - CUDA 11.7](https://pytorch.org/get-started/previous-versions/#linux-and-windows-2).

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- CPU only:

[or CPU only binaries](https://pytorch.org/get-started/previous-versions/#linux-and-windows-2).

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

Optional dependencies for data generation:

```bash
conda install clawpack jax jaxlib python-dotenv
```

## 📖 Overview

### Motivation

Neural operators are powerful surrogates for PDE simulations but often fail under distribution shifts due to a lack of fundamental physics priors. Numerical solvers preserve conservation and symmetries, but data-driven models typically do not.

### Our Contribution

We propose multiphysics joint training that combines:

- The original PDE simulations (e.g., Diffusion–Reaction, Navier–Stokes)
- Their decomposed basic forms (e.g., pure diffusion, pure convection)

This simple but principled approach:

- Encodes physics priors without extra cost
- Enables >11.5% improvement in nRMSE
- Generalizes to OOD and real-world (ScalarFlow) scenarios

## 📂 Repository Structure

```bash
├── Hyena Model Comparison/          # Experiments with Hyena neural operator
├── OFormer Model Comparison/        # Experiments with OFormer operator
├── pdebench/                        # Core PDEBench code
│   ├── data_gen/                    # Data generation utilities
│   │   ├── configs/                 # PDE configs (YAML)
│   │   ├── src/                     # PDE solvers & utilities
│   │   ├── gen_diff_react.py        # Generate Diffusion-Reaction data
│   │   ├── gen_ns_incomp.py         # Generate Navier-Stokes data
│   │   ├── run_sim.sh               # Script for running simulations
│   │   └── velocity2vorticity.py    # Helper conversion tool
│   ├── models/                      # Neural operator architectures
│   │   ├── fno/                     # Fourier Neural Operator (baseline)
│   │   ├── fno_aux/                 # FNO + auxiliary training (ours)
│   │   ├── Transformer_2D_DR/       # Transformer for Diffusion–Reaction
│   │   ├── Transformer_2D_NS/       # Transformer for 2D Navier–Stokes
│   │   ├── Transformer_3D_NS/       # Transformer for 3D Navier–Stokes
│   │   ├── train_models_forward.py  # Training entrypoint
│   │   ├── train_models_aux_forward.py
│   │   ├── analyse_result_forward.py
│   │   └── metrics.py               # Evaluation metrics
├── Plot Generator/                  # Scripts for generating paper figures
│   ├── 2D_DR_plot.py
│   ├── 2D_NS_plot.py
│   ├── 3D_NS_Vis.py
│   └── rollout.py
├── Rollout Experiment/              # Autoregressive rollout experiments
│   ├── 2D_NS_Baseline_rollout/
│   ├── 2D_NS_Ours_rollout/
│   └── *.png                        # Example visualization outputs
├── LICENSE.txt
├── pyproject.toml
├── noxfile.py
└── README.md


```

## 🚀 Usage

### 1. Generate Data

All PDE datasets are generated via pdebench/data_gen:

```bash
cd pdebench/data_gen
bash run_sim.sh

```

or directly with Python:

```bash
python gen_diff_react.py --config configs/diff-react.yaml
python gen_ns_incomp.py --config configs/ns_incomp.yaml

```

This produces training datasets under the configured directories.

### 2. Train Models

- Baseline FNO:

```bash
python -m pdebench.models.fno.train --config config/fno.yaml

```

- FNO + Fundamental Physics (ours):

```bash
python -m pdebench.models.fno_aux.fno_train_aux --config config/fno_aux.yaml

```

- Transformer (2D Navier–Stokes):

```bash
python -m pdebench.models.Transformer_2D_NS.train_models_forward --config config/transformer_ns.yaml

```

Training logs and checkpoints will be saved under models/<architecture>/<experiment>.

### 3. Evaluate & Rollout

Evaluate trained models:

```bash
python -m pdebench.models.analyse_result_forward --checkpoint path/to/model.pt

```

Run rollout experiments (autoregressive inference):

```bash
cd Rollout\ Experiment
python tme.py

```

Visualizations:

```bash
python Plot\ Generator/2D_NS_plot.py
python Plot\ Generator/3D_NS_Vis.py

```

## 📊 Reproducing Paper Results

- Data Efficiency → train_models_forward.py + subsampling configs
- Long-term Consistency → Rollout Experiment/ + rollout.py
- OOD Generalization → configs in pdebench/data_gen/configs with shifted parameters
- Synthetic-to-Real → compare against ScalarFlow dataset (scripts in 3D_NS_Vis.py)
- Figures → generated via Plot Generator/

We follow the same hyperparameters as reported in the paper (Appendix B).

## 📝 Citation

If you use this repository, please cite:

```bash
@inproceedings{neurips2025_fundamental_physics,
  title     = {Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge},
  author    = {Anonymous},
  booktitle = {Proceedings of the 39th Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}

```

## ⚠️ Limitations

- Current PDE coverage: Diffusion–Reaction, 2D/3D Incompressible Navier–Stokes
- Decomposition strategies: limited to dropping reaction/pressure/diffusion terms
- Architectures: evaluated on FNO and Transformer only

Future work will expand to more PDE systems, architectures, and decomposition strategies.

## 📜 License

Except where otherwise stated this code is released under the MIT license.

    Copyright 2022 NEC Labs Europe GmbH, Stuttgart University, CSIRO and PDEbench contributors

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
