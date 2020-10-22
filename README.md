<p align="center">
  <img width="231" height="271.5" src="https://github.com/tgac-vumc/BLADE/blob/master/logo_final_small.png">
</p>

# BLADE: Bayesian Log-normAl DEconvolution


BLADE (Bayesian Log-normAl DEconvolution) was designed to jointly estimate cell type composition and gene expression profiles per cell type in a single-step while accounting for the observed gene expression variability in single-cell RNA-seq data. 

Demo notebook is available [here](https://github.com/tgac-vumc/BLADE/blob/master/jupyter/BLADE%20-%20Demo%20script.ipynb).

<p align="center">
  <img width="490" height="820" src="https://github.com/tgac-vumc/BLADE/blob/master/framework.png">
</p>


## Installation

### Using Conda
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

You can install all the necessary dependency using the following command.

```
conda install mamba -c conda-forge -y
mamba create --name BLADE -c conda-forge -c bioconda jupyter numpy numba scikit-learn joblib multiprocess time scipy qgrid seaborn
```

Then, the `BLADE` environment can be activate by:

```
conda activate BLADE
```

### Using Singularity

If you have Singularity, you can simply pull the singularity container with all dependency resolved.

```
singularity pull shub://tgac-vumc/BLADE
```

