# Quantum regression with a problem-specific ansatz

This repository contains the Python code that generates the datasets and reproduces the results presented in our paper [*<title>*](https://arxiv.org/abs/XXXX.XXXXX).

## Overview

The project is organized into two main parts, corresponding to the two learning tasks in the paper:

- **Entanglement** — predicting the entanglement of bipartite qubit states.
- **Ising model** — predicting the relative field strength of the transverse-field Ising model.

## Repository structure

### `Entanglement/`

Predicting entanglement of bipartite qubit states. Contains four subfolders:

- **`Pure&Iso_states/`** — predicting entanglement of pure and isotropic states using two copies of a bipartite qubit state.
- **`10classes/`** — predicting entanglement of mixed states using four copies of the input state, with ten-class Hermitian operators used as the ansatz.
- **`Pauli/`** — predicting entanglement of mixed states using four copies of the input state, with *k*-local Pauli strings used as the ansatz.
- **`function_learning/`** — various versions of the ten-class functions derived from the ten-class Hermitian ansatz, obtained by expanding or trimming the function set. We conclude that the ten-class functions from the ten-class Hermitian classes are the optimal choice.

### `Ising_model/`

Selecting the best set of Pauli strings that respect the symmetry of the 8-qubit Hamiltonian, achieving accuracy comparable to — and lower variance than — training on the full set of Pauli strings.

- **`Results/`** — uses the pre-trained parameters to construct the optimal observable for predicting the relative field strength of the transverse-field Ising model, and to reproduce the figures in the paper.

## Requirements

The code was tested with the following versions:

| Package    | Version |
|------------|---------|
| python     | 3.11    |
| numpy      | 2.4.6   |
| scipy      | 1.17.1  |
| matplotlib | 3.11.0  |
| qutip      | 5.3.0   |
| cvxpy      | 1.9.1   |

The most reliable way to reproduce this environment is a dedicated conda environment built from conda-forge:

```bash
conda create -n qregress -c conda-forge python=3.11 numpy scipy matplotlib qutip cvxpy jupyter ipykernel
conda activate qregress
```

> **Note (Windows / BLAS).** On some Windows + conda setups, the default MKL BLAS backend causes a hard crash on complex matrix multiplication (`Windows fatal exception: code 0xc06d007f`). If you encounter this, switch to the OpenBLAS backend:
> ```bash
> conda install -c conda-forge "libblas=*=*openblas" numpy
> `
