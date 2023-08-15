# Python Environment

## Conda Create and Activate Environment

```
conda create --name MolTran_CUDA11 python=3.8.10
conda activate MolTran_CUDA11
```

## Conda Install Packages
```
conda install pytorch==1.7.1 cudatoolkit=11.0 -c pytorch
conda install numpy=1.22.3 pandas=1.2.4 scikit-learn=0.24.2 scipy=1.6.2
conda install rdkit==2022.03.2 -c conda-forge
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
git checkout tags/22.03 -b v22.03
export CUDA_HOME='Cuda 11 install'
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
