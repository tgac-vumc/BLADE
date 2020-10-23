<p align="center">
  <img width="254" height="281" src="https://github.com/tgac-vumc/BLADE/blob/master/logo_final_small.png">
</p>

# BLADE: Bayesian Log-normAl DEconvolution


BLADE (Bayesian Log-normAl DEconvolution) was designed to jointly estimate cell type composition and gene expression profiles per cell type in a single-step while accounting for the observed gene expression variability in single-cell RNA-seq data. 

Demo notebook is available [here](https://github.com/tgac-vumc/BLADE/blob/master/jupyter/BLADE%20-%20Demo%20script.ipynb).

<p align="center">
  <img width="100%" height="100%" src="https://github.com/tgac-vumc/BLADE/blob/master/framework.png">
</p>


BLADE framework. To construct a prior knowledge of BLADE, we used single-cell sequencing data. Cell are subject to phenotyping, clustering and differential gene expression analysis. Then, for each cell type, we retrieve average expression profiles (red cross and top heatmap), and standard deviation per gene (blue circle and bottom heatmap). This prior knowledge is then used in the hierarchical Bayesian model (bottom right) to deconvolute bulk gene expression data.


## Installation

### Using PiP

The python package of BLADE is available on PiP.
You can simply:

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

You can install all the necessary dependency using the following command.

```
conda env create --file environment.yml
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

