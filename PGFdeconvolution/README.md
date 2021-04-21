

# Generic deconvolution method with known cellular compositions

We developed a simple, generic maximum-likelihood based convolution model. 
The method is based on a probability generating function (PGF) that approximate convoluted random variable.

The R scripts in this folder allows users to compare log-normal (LN) convolution and Negative binomial (NB) convolution models to perform deconvolution on real TCGA data.

- `install.R`: A script to install dependencies.
- `source_estmusigmadist2.R`: core functions for PGF deconvolution
- `LogNormalvsNB_TCGA.R`: A script to compare Negative Binomial and Log-normal distribution for deconvolution.
- `LogNormalvsNB_TCGA.html`: A knitted html report generated from `LogNormalvsNB_TCGA.R`
- `TCGA-MESO.RData`/`TCGA-SARC.RData`: Gene expression data of two TCGA cohorts (mesothelioma and sarcoma)
