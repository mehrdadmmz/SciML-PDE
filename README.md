The code repository for the NeurIPS 2025 paper: *Learning Data-Efficient and Generalizable Neural Operators via Fundamental Physics Knowledge*


## Installation using pip

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


## Installation using conda:

If you like you can also install dependencies using anaconda, we suggest to use
[mambaforge](XXXX) as a
distribution. Otherwise you may have to **enable the conda-forge** channel for
the following commands.

Starting from a fresh environment:

```
conda create -n myenv python=3.9
conda activate myenv
```

Install dependencies for model training:

```
conda install deepxde hydra-core h5py -c conda-forge
```

According to your hardware availability, either install PyTorch with CUDA
support:

- [see previous-versions/#linux - CUDA 11.7](XXXX).

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- [or CPU only binaries](XXXX).

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
```

Optional dependencies for data generation:

```
conda install clawpack jax jaxlib python-dotenv
```

