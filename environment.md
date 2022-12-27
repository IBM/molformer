# Python Environment 

## Conda Create and Activate Environment

```
conda create --name MolTran_CUDA11 python=3.8.10
conda activate MolTran_CUDA11
```

## Conda Install Packages
```
conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch
conda install rdkit==2021.03.2 pandas=1.2.4 scikit-learn=0.24.2 scipy=1.6.3 
```

## Pip install Packages
```
pip install transformers==4.6.0 pytorch-lightning==1.1.5 pytorch-fast-transformers==0.4.0 datasets==1.6.2 jupyterlab==3.4.0 ipywidgets==7.7.0 bertviz==1.4.0
```

## Compile Apex from source

Due to the use of [Apex Optimizers](https://nvidia.github.io/apex/optimizers.html), Apex must be compiled with CUDA and C++ extensions via


```
git clone https://github.com/NVIDIA/apex
cd apex
export CUDA_HOME='Cuda 11 install'
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
