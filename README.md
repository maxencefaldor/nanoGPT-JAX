# nanoGPT in JAX !

This repository is based on [nanoGPT](https://github.com/karpathy/nanoGPT) from [karpathy](https://github.com/karpathy) and you will find:
- ``nanoGPT.ipynb``, the original notebook in PyTorch
- ``nanoGPT_jax.ipynb``, the original notebook translated in JAX
- ``nanoGPT_jax.py``, a script to train a nanoGPT of ~200 lines in JAX

You can find an example of the text it generates after training it for ~30 minutes in ``outputs/output.txt``.

## Installation

To run this code, you need to clone the repository and install the required libraries with:
```bash
git clone ...
pip install -r requirements.txt
```

## Apptainer

To build a container using Apptainer make sure you are in the root of the repository and then run:

```bash
apptainer build --fakeroot --force apptainer/container.sif apptainer/container.def
```
