Bootstrap: yum
OSVersion: 7
MirrorURL: http://mirror.centos.org/centos-7/7/os/x86_64/
Include: yum

%environment    
    export PATH=/usr/local/bin:$PATH

%post
    ./environment

    yum -y update
    yum -qq -y install curl tar bzip2 git zip
    curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -bfp /usr/local
    conda update conda -y
    
    conda install mamba -c conda-forge -y
    mamba install -c conda-forge -c bioconda jupyter numpy numba scikit-learn joblib multiprocess time scipy qgrid seaborn
    pip install -y BLADE-Deconvolution==0.0.4
    jupyter nbextension enable --py --sys-prefix qgrid
    jupyter nbextension enable --py --sys-prefix widgetsnbextension
 


    rm -rf /tmp/miniconda.sh
