<p align="center">
  <img width="254" height="281" src="https://github.com/tgac-vumc/BLADE/blob/master/logo_final_small.png">
</p>

# BLADE: Bayesian Log-normAl DEconvolution


[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI version](https://badge.fury.io/py/BLADE-Deconvolution.svg)](https://badge.fury.io/py/BLADE-Deconvolution)
[![https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg](https://www.singularity-hub.org/static/img/hosted-singularity--hub-%23e32929.svg)](https://singularity-hub.org/collections/4861)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/tgac-vumc/BLADE/master)



BLADE (Bayesian Log-normAl DEconvolution) was designed to jointly estimate cell type composition and gene expression profiles per cell type in a single-step while accounting for the observed gene expression variability in single-cell RNA-seq data.

<p align="center">
  <img width="100%" height="100%" src="https://github.com/tgac-vumc/BLADE/blob/master/framework.png">
</p>


BLADE framework. To construct a prior knowledge of BLADE, we used single-cell sequencing data. Cell are subject to phenotyping, clustering and differential gene expression analysis. Then, for each cell type, we retrieve average expression profiles (red cross and top heatmap), and standard deviation per gene (blue circle and bottom heatmap). This prior knowledge is then used in the hierarchical Bayesian model (bottom right) to deconvolute bulk gene expression data.

#### Demo notebook is available [here](https://github.com/tgac-vumc/BLADE/blob/master/jupyter/BLADE%20-%20Demo%20script.ipynb). You can also run the demo using [Binder](https://mybinder.org/v2/gh/tgac-vumc/BLADE/master). 
Note that for the testing on Binder, parallel processing has to be disabled by setting `Njob` to 1. BLADE significantly performs better with high number of cores, epecially when `Nsample`, `Ngene` and `Ncell` is high. In case of Binder, we recommend the following setting:

- `Ncell=3`
- `Ngene=50`
- `Nsample=10`

It takes about 30 minutes to complete the demo execution on Binder.


## System Requirements

### Hardware Requirements

BLADE can run on the minimal computer spec, such as Binder (1 CPU, 2GB RAM on Google Cloud), when data size is small. However, BLADE can significantly benefit from the larger amount of CPUs and RAM. Empirical Bayes procedure of BLADE runs independent optimization procedure that can be parallelized. In our evaluation, we used a computing node with the following spec:

- 40 threads (Xeon 2.60GHz)
- 128 GB RAM


### OS Requirements

The package development version is tested on Linux operating systems. (CentOS 7 and Ubuntu 16.04). 


## Installation

### Using pip

The python package of BLADE is available on pip.
You can simply (takes only <1min):

```
pip install BLADE_Deconvolution
```

We tested BLADE with `python => 3.6`.


### Using Conda

One can create a conda environment contains BLADE and also other dependencies to run [Demo](https://github.com/tgac-vumc/BLADE/blob/master/jupyter/BLADE%20-%20Demo%20script.ipynb).
The environment definition is in [environment.yml](https://github.com/tgac-vumc/BLADE/environment.yml).

### Step 1: Installing Miniconda 3
First, please open a terminal or make sure you are logged into your Linux VM. Assuming that you have a 64-bit system, on Linux, download and install Miniconda 3 with:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
On MacOS X, download and install with:

```
curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

### Step 2: Create a conda environment

You can install all the necessary dependency using the following command (may takes few minutes).

```
conda env create --file environment.yml
```

Then, the `BLADE` environment can be activate by:

```
conda activate BLADE
```

### Using Singularity

If you have Singularity, you can simply pull the singularity container with all dependency resolved (in few minutes, depends on the network speed).

```
singularity pull shub://tgac-vumc/BLADE
```

## Overview of BLADE

In the BLADE package, you can load the following functions and modules.

- `BLADE`: A class object contains core algorithms of `BLADE`. Users can reach internal variables (`Nu`, `Omega`, and `Beta`) and functions for calculating objective functions (ELBO function) and gradients with respect to the variational parameters. There also is an optimization function (`BLADE.Optimize()`) for performing L-BFGS optimization. Though this is the core, we also provide a more accessible function (`BLADE_framework`) that performs deconvolution. See below to obtain the current estimate of cellualr fractions, gene expression profiles per cell type and per sample:
  - `ExpF(self.Beta)` : returns a `Nsample` by `Ngene` matrix contains estimated fraction of each cell type in each sample.
  - `self.Nu`: a `Nsample` by `Ngene` by `Ncell` multidimensional array contains estimated gene expression levels of each gene in each cell type for each sample.
  - `numpy.mean(self.Nu,0)`: To obtain a estimated gene expression profile per cell type, we can simply take an average across the samples.

- `Framework`: A framework based on the `BLADE` class module above. Users need to provide the following input/output arguments.
  - Input arguments
    - `X`: a `Ngene` by `Ncell` matrix contains average gene expression profiles per cell type (a signature matrix) in log-scale.
    - `stdX`: a `Ngene` by `Ncell` matrix contains standard deviation per gene per cell type (a signature matrix of gene expression variability).
    - `Y`: a `Ngene` by `Nsample` matrix contains bulk gene expression data. This should be in linear-scale data without log-transformation.
    - `Ind_Marker`: Index for marker genes. By default, `[True]*Ngene` (all genes used without filtering). For the genes with `False` they are excluded in the first phase (Empirical Bayes) for finidng the best hyperparameters.
    - `Ind_sample`: Index for the samples used in the first phase (Empirical Bayes). By default, `[True]*Nsample` (all samples used).
    - `Alphas`, `Alpha0s`, `Kappa0s` and `SYs`: all possible hyperparameters considered in the phase of Empirical Bayes. A default parameters are offered as described in the manuscript (to appear): `Alphas=[1,10]`, `Alpha0s=[0.1, 1, 5]`, `Kappa0s=[1,0.5,0.1]` and `SYs=[1,0.3,0.5]`. 
    - `Nrep`: Number of repeat for evaluating each parameter configuration in Empirical Bayes phase. By default, `Nrep=3`.
    - `Nrepfinal`: Number of repeated optimizations for the final parameter set. By default, `Nrepfinal=10`.
    - `Njob`: Number of jobs executed in parallel. By default, `Njob=10`.
  - Output values
    - `final_obj`: A final `BLADE` object with optimized variational parameters and hyperparameters.
    - `best_obj`: The best object form Empirical Bayes step. If no genes and samples are filtered, `best_obj` is the same as `final_obj`.
    - `best_set`: A list contains the hyperparameters selected in the Empirical Bayes step.
    - `All_out`: A list of `BLADE` objects from the Empirical Bayes step.
- `BLADE_job`/`Optimize`: Internal functions used by `Framework`.
