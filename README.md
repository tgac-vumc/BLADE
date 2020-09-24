# BLADE (when logo is ready, put it here)
BLADE: Bayesian Log-normAl DEconvolution for enhanced in silico microdissection of bulk gene expression data

BLADE was designed to jointly estimate cell type composition and gene expression profiles per cell type in a single-step while accounting for the observed gene expression variability in single-cell RNA-seq data. 

<p align="center">
  <img width="490" height="820" src="https://github.com/tgac-vumc/BLADE/blob/master/framework.png">
</p>


## Usage

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

### Step 2: Downloading repository & creating environment

```
mkdir BLADE
cd BLADE
git clone https://github.com/tgac-vumc/BLADE
conda env create --name BLADE --file env.yaml
```

### Using Singularity (NEEDS TO BE DONE)

The singularity container holds a virtual environment of X and it's available with:
```
singularity pull shub://tgac-vumc/BLADE
```
