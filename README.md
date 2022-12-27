This repository provides pytorch source code, and data associated with our Nature Machine Intelligence _(10.1038/s42256-022-00580-7)_ publication, "Large-Scale Chemical Language Representations Capture Molecular Structure and Properties".

Paper: [NMI Link](https://rdcu.be/c12D0) / [Arxiv Link](https://arxiv.org/abs/2106.09553)

# MoLFormer

**MoLFormer** is a large-scale chemical language model designed with the intention of learning a model trained on small molecules which are represented as SMILES strings. MoLFormer leverges Masked Language Modeling and employs a linear attention Transformer combined with rotary embeddings.

![MoLFormer](https://media.github.ibm.com/user/4935/files/594363e6-497b-4b91-9493-36ed46f623a2)

An overview of the MoLFormer pipeline is seen in the image above. One can see that the transformer based neural network model is trained on a large collection of chemical molecules represented by SMILES sequences from two public chemical datasets PubChem and Zinc in a self-supervised fashion.  The MOLFORMER architecture was designed with an efficient linear attention mechanism and relative positional embeddings with the goal of learning a meaningful and compressed representation of chemical molecules. After training the MOLFORMER foundation model was then adopted to different downstream molecular property prediction tasks via fine-tuning on task-specific data. To further test the representative power of MOLFORMER the MOLFORMER encodings were used to recover molecular similarity, and analysis on the correspondence between the interatomic spatial distance and attention value for a given molecule was performed.

1. [Getting Started](#getting-started)
    1. [Pretrained Models and training logs](#pretrained-models-and-training-logs)
    2. [Replicating Conda Environment](#replicating-conda-environment)
3. [Data](#data)
    1. [PreTraining Datasets](#pretraining-datasets)
    2. [Finetuning Datasets](#finetuning-datasets)
5. [PreTraining](#pretraining)
6. [FineTuning](#finetuning)
7. [Attention Visualization Analysis](#attention-visualization-analysis)
8. [Citations](#citatiobs)


## Getting Started

**This Code and Environment have been tested on Nvidia V100s**

#### Pretrained Models and training logs
If Training from scratch the resulting Pretrained models and associated training logs will be located in the /data directory in the following hierarchy. 

```
data/
├── checkpoints
|   └── linear_model.ckpt
|   └── full_model.ckpt
├── Full_Attention_Rotary_Training_Logs
│   ├── events.out.tfevents.1628698179.cccxc544.604661.0
│   └── hparams.yaml
└── Linear_Rotary_Training_Logs
    ├── events.out.tfevents.1620915522.cccxc406.63025.0
    └── hparams.yaml
```

We are providing checkpoints of a MoLFormer model pre-trained on a dataset of ~100M molecules. This dataset combines 10% of Zinc and 10% of PubChem molecules used for MoLFormer-XL training. The accompanying pre-trained model shows competitive performance on classification and regression benchmarks from MoleculeNet. (see Extended data Tables 1-2 in [https://arxiv.org/abs/2106.09553](https://arxiv.org/abs/2106.09553)). These checkpoints are available at (https://ibm.box.com/v/MoLFormer-data](https://ibm.box.com/v/MoLFormer-data)

#### Replicating Conda Environment 

Due to the use of apex.optimizers in our code, Apex must be compiled from source. Step-by-step directions are provided in [environment.md](environment.md)

## Data

Datasets are available at [https://ibm.box.com/v/MoLFormer-data](https://ibm.box.com/v/MoLFormer-data)

### PreTraining Datasets
Due to the large nature of the combination of the PubChem and Zinc (over 1.1 billion molecules in total) datasets the code expects the data to be in a certain location and format.  The details of the of this processing is documented below for each individaul dataset. 

The code expects both the zinc15(ZINC) and pubchem datasets to be located in ```./data/``` directory of the training diretory. 
  * Zinc15 itself should be in located ```data/ZINC/``` and is expected to be processed in multiple smi files which contains one smiles string per line.
  * PubChem should be located in ```data/pubchem/``` and is expected to be processed as a single “CID-SMILES” text file with 2 columns (index and smiles string).  We took the raw Pubchem dataset and converted every smiles molecule into the canonical form, utilizing rdkit, as well as trimmed down the file itself. Our dataloader expects Pubchem to be in our converted form and will not run on the raw pubchem file. 

```
data/
├── pubchem
│   └── CID-SMILES-CANONICAL.smi
└── ZINC
    ├── AAAA.smi
    ├── AAAB.smi
    ├── AAAC.smi
    ├── AAAD.smi
    ├── AABA.smi
    ├── AABB.smi
    ├── AABD.smi
    ├── AACA.smi
    ├── AACB.smi
    ├── AAEA.smi
    ├── AAEB.smi
    ├── AAED.smi
    ├── ABAA.smi
    ├── ABAB.smi
    ├── ABAC.smi
    ├── ABAD.smi
    ├── ABBA.smi
    ├── ABBB.smi
    ├── ABBD.smi
    ├── ABCA.smi
    ├── ABCB.smi
    ├── ABCD.smi
    ├── ABEA.smi
    ├── ABEB.smi
    ├── ABEC.smi
    ├── ABED.smi
    ├── ACAA.smi
    ├── ACAB.smi
```

### Finetuning Datasets
Just as with the pretraining data the code expects the finetuning datasets to be in the following hierarchy. These datasets were provided in the finetune_datasets.zip

```
data/
├── bace
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── bbbp
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── clintox
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── esol
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── freesolv
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── hiv
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
├── lipo
│   ├── lipo_test.csv
│   ├── lipo_train.csv
│   └── lipo_valid.csv
├── qm9
│   ├── qm9.csv
│   ├── qm9_test.csv
│   ├── qm9_train.csv
│   └── qm9_valid.csv
├── sider
│   ├── test.csv
│   ├── train.csv
│   └── valid.csv
└── tox21
    ├── test.csv
    ├── tox21.csv
    ├── train.csv
    └── valid.csv
```


## Pretraining
For pre-training we use the masked language model method to train the model from scratch.

MoLFormer is pre-trained on canonicalized SMILES of >1 B molecules from ZINC and PubChem with the following constraints:

During pre-processing, the compounds are filtered to keep a maximum length of 211 characters. A 100/0/0 split was used for training, validation, and test, i.e. we used all the data for training the model. As a confidence test we would evaluate the model at the end of each epoch on the following data (find the data we used for eval). Data canonicalization was performed using RDKit.

The pre-training code provides an example of data processing and training of a model trained on a smaller pre-training dataset size, which requires 16 v100 GPUs. The remainder of this README contains an installation guide for this repo, descriptions and links to pre-training and fine-tuning datasets, configuration files and python codes for model pre-training and fine-tuning, and jupyter notebook for attention map visualization and analysis for a given molecule. A MoLFormer instance pre-trained on xxx data is also provided.

To train a model run: 

> bash run_pubchem_light.sh

## Finetuning

The finetuning related dataset and environment can be found in [finetuning datasets](finetuning_datasets) and [environment.md](environment.md) respectively. Once you have the environment set up, you can run a fine-tune task by running

> bash run_finetune_mu.sh

Finetuning training/checkpointing resources will be available in the diretory named ```checkpoint_<measure_name>```. The path to the results csv will be in the form of ```./checkpoint_<measure_name>/<measure_name>/results/results_.csv``` The ```results_.csv``` file contains 4 columns of data. Column one contains the validation score for each epoch while column 2 contains the test score for each epoch. Column 3 contains the best validation score observed up to that point of fine tuning while column 4 is the test score of the epoch which had the best validation score.

## Attention Visualization Analysis

The `notebooks` directory provide attention visualization for two setup with Rotary Embeddings:
- **Linear attention** (./notebooks/full_attention_rotary/attention_analysis_rotary_full.ipynb)
- **Full attention** (./notebooks/linear_attention_rotary/attention_analysis_rotary_linear.ipynb)

The checkpoints required for the above models are to be placed in `./data/checkpoints`

## Citations
```
@article{10.1038/s42256-022-00580-7, 
year = {2022}, 
title = {{Large-scale chemical language representations capture molecular structure and properties}}, 
author = {Ross, Jerret and Belgodere, Brian and Chenthamarakshan, Vijil and Padhi, Inkit and Mroueh, Youssef and Das, Payel}, 
journal = {Nature Machine Intelligence}, 
doi = {10.1038/s42256-022-00580-7}, 
abstract = {{Models based on machine learning can enable accurate and fast molecular property predictions, which is of interest in drug discovery and material design. Various supervised machine learning models have demonstrated promising performance, but the vast chemical space and the limited availability of property labels make supervised learning challenging. Recently, unsupervised transformer-based language models pretrained on a large unlabelled corpus have produced state-of-the-art results in many downstream natural language processing tasks. Inspired by this development, we present molecular embeddings obtained by training an efficient transformer encoder model, MoLFormer, which uses rotary positional embeddings. This model employs a linear attention mechanism, coupled with highly distributed training, on SMILES sequences of 1.1 billion unlabelled molecules from the PubChem and ZINC datasets. We show that the learned molecular representation outperforms existing baselines, including supervised and self-supervised graph neural networks and language models, on several downstream tasks from ten benchmark datasets. They perform competitively on two others. Further analyses, specifically through the lens of attention, demonstrate that MoLFormer trained on chemical SMILES indeed learns the spatial relationships between atoms within a molecule. These results provide encouraging evidence that large-scale molecular language models can capture sufficient chemical and structural information to predict various distinct molecular properties, including quantum-chemical properties. Large language models have recently emerged with extraordinary capabilities, and these methods can be applied to model other kinds of sequence, such as string representations of molecules. Ross and colleagues have created a transformer-based model, trained on a large dataset of molecules, which provides good results on property prediction tasks.}}, 
pages = {1256--1264}, 
number = {12}, 
volume = {4}
}
```

```
@misc{https://doi.org/10.48550/arxiv.2106.09553,
  doi = {10.48550/ARXIV.2106.09553},
  url = {https://arxiv.org/abs/2106.09553},
  author = {Ross, Jerret and Belgodere, Brian and Chenthamarakshan, Vijil and Padhi, Inkit and Mroueh, Youssef and Das, Payel},
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), Biomolecules (q-bio.BM), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Biological sciences, FOS: Biological sciences},
  title = {Large-Scale Chemical Language Representations Capture Molecular Structure and Properties},
  publisher = {arXiv},
  year = {2021},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
