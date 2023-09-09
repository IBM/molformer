
import argparse
import glob
import torch
import shutil
import rdkit
from torch import nn
# import args
from molformer.model.args import get_parser as ARGS
import os
import numpy as np
import random
import getpass
from datasets import load_dataset, concatenate_datasets
from pubchem_encoder import Encoder
import lightning.pytorch as pl
# from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from torch.utils.data import DataLoader
import subprocess
# from pytorch_lightning import LightningModule
from molformer.model.base_bert import LightningModule

class MoleculeModule(pl.LightningDataModule):
    def __init__(self,  max_len, data_path, train_args):
        super().__init__()
        self.data_path = data_path
        self.train_args = train_args  # dict with keys {'batch_size', 'shuffle', 'num_workers', 'pin_memory'}
        self.text_encoder = Encoder(max_len)

    def prepare_data(self):
        pass

    def get_vocab(self):
        #using home made tokenizer, should look into existing tokenizer
        return self.text_encoder.char2id

    def get_cache(self):
        return self.cache_files
    def setup(self, stage=None):
        #using huggingface dataloader
        # create cache in tmp directory of locale mabchine under the current users name to prevent locking issues
        pubchem_path = {'train':'../../../data/pubchem/CID-SMILES-CANONICAL.smi'}
        if 'CANONICAL' in pubchem_path:
            pubchem_script = '/home/mi52qep/proj/molformer_refactor/src/molformer/model/pubchem_canon_script.py'
        else:   
            pubchem_script = '/home/mi52qep/proj/molformer_refactor/src/molformer/model/pubchem_script.py'
        # zinc_path = '../../../data/ZINC'
        # if 'ZINC' in self.data_path or 'zinc' in self.data_path:
        #     zinc_files = [f for f in glob.glob(os.path.join(zinc_path,'*.smi'))]
        #     for zfile in zinc_files:
        #         print(zfile)
        #     self.data_path = {'train': zinc_files}
        #     dataset_dict = load_dataset('./zinc_script.py', data_files=self.data_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'zinc'),split='train')

        # if 'pubchem' in self.data_path:
        dataset_dict =  load_dataset(pubchem_script, data_files=pubchem_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'pubchem'), split='train')
        # elif 'both' in self.data_path or 'Both' in self.data_path or 'BOTH' in self.data_path:
        #     dataset_dict_pubchem =  load_dataset(pubchem_script, data_files=pubchem_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'pubchem'),split='train')
        #     zinc_files = [f for f in glob.glob(os.path.join(zinc_path,'*.smi'))]
        #     for zfile in zinc_files:
        #         print(zfile)
        #     self.data_path = {'train': zinc_files}
        #     dataset_dict_zinc =  load_dataset('./zinc_script.py', data_files=self.data_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'zinc'),split='train')
        #     dataset_dict = concatenate_datasets([dataset_dict_zinc, dataset_dict_pubchem])
        self.pubchem= dataset_dict
        print(dataset_dict.cache_files)
        self.cache_files = []

        for cache in dataset_dict.cache_files:
            tmp = '/'.join(cache['filename'].split('/')[:4])
            self.cache_files.append(tmp)


    def train_dataloader(self):
        loader =  DataLoader(self.pubchem, collate_fn=self.text_encoder.process, **self.train_args)
        print(len(loader))
        return loader

    # def val_dataloader(self):
    #     pass
    
    # def test_dataloader(self):
    #     pass
    
class CheckpointEveryNSteps(pl.Callback):
    """
        Save a checkpoint every N steps, instead of Lightning's default that checkpoints
        based on validation loss.
    """

    def __init__(self, save_step_frequency=-1,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
        ):
        """
        Args:
        save_step_frequency: how often to save in steps
        prefix: add a prefix to the name, only used if
        use_modelcheckpoint_filename=False
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0 and self.save_step_frequency > 10:

            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)
            
            
class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)
            
            
@rank_zero_only
def remove_tree(cachefiles):
    if type(cachefiles) == type([]):
        #if cachefiles are identical remove all but one file path
        cachefiles = list(set(cachefiles))
        for cache in cachefiles:
            shutil.rmtree(cache)
    else:
        shutil.rmtree(cachefiles)


def get_nccl_socket_ifname():
    ipa = subprocess.run(['ip', 'a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ipa.stdout.decode('utf-8').split('\n')
    all_names = []
    name = None
    for line in lines:
        if line and not line[0] == ' ':
            name = line.split(':')[1].strip()
            continue
        if 'link/infiniband' in line:
            all_names.append(name)
    os.environ['NCCL_SOCKET_IFNAME'] = ','.join(all_names)


def fix_infiniband():

    get_nccl_socket_ifname()
    os.environ['NCCL_IB_CUDA_SUPPORT'] = '1'
    ibv = subprocess.run('ibv_devinfo', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = ibv.stdout.decode('utf-8').split('\n')
    exclude = ''
    for line in lines:
        if 'hca_id:' in line:
            name = line.split(':')[1].strip()
        if '\tport:' in line:
            port = line.split(':')[1].strip()
        if 'link_layer:' in line and 'Ethernet' in line:
            exclude = exclude + f'{name}:{port},'
    if exclude:
        exclude = '^' + exclude[:-1]
        # print(exclude)
        os.environ['NCCL_IB_HCA'] = exclude


from attributedict.collections import AttributeDict
from molformer.utils import get_argparse_defaults

def main():    
    fix_infiniband()
    config = AttributeDict(get_argparse_defaults(ARGS()))
    config.num_nodes = 1
    config.n_batch = 250
    config.n_head = 12
    config.n_layer = 12
    config.n_embd = 768
    config.max_len = 202
    config.d_dropout = 0.2
    config.lr_start = 3e-5
    config.lr_multiplier = 8
    config.n_workers = 8
    config.max_epochs = 4
    config.gpu = 1
    config.num_nodes = 1
    config.accelerator = 'ddp' 
    config.num_feats = 32
    config.root_dir = './training_runs'
    config.checkpoint_every = 10
    config.train_load = 'both'
    
    if config.num_nodes > 1:
        # print("Using " + str(config.num_nodes) + " Nodes----------------------------------------------------------------------")
        LSB_MCPU_HOSTS = os.environ["LSB_MCPU_HOSTS"].split(' ') # Parses Node list set by LSF, in format hostname proceeded by number of cores requested
        HOST_LIST = LSB_MCPU_HOSTS[::2] # Strips the cores per node items in the list
        os.environ["MASTER_ADDR"] = HOST_LIST[0] # Sets the MasterNode to thefirst node on the list of hosts
        os.environ["MASTER_PORT"] = "54966"
        os.environ["NODE_RANK"] = str(HOST_LIST.index(os.environ["HOSTNAME"])) #Uses the list index for node rank, master node rank must be 0
        #os.environ["NCCL_SOCKET_IFNAME"] = 'ib,bond'  # avoids using docker of loopback interface
        os.environ["NCCL_DEBUG"] = "INFO" #sets NCCL debug to info, during distributed training, bugs in code show up as nccl errors
        #os.environ["NCCL_IB_CUDA_SUPPORT"] = '1' #Force use of infiniband
        #os.environ["NCCL_TOPO_DUMP_FILE"] = 'NCCL_TOP.%h.xml'
        #os.environ["NCCL_DEBUG_FILE"] = 'NCCL_DEBUG.%h.%p.txt'
        print(os.environ["HOSTNAME"] + " MASTER_ADDR: " + os.environ["MASTER_ADDR"])
        print(os.environ["HOSTNAME"] + " MASTER_PORT: " + os.environ["MASTER_PORT"])
        print(os.environ["HOSTNAME"] + " NODE_RANK " + os.environ["NODE_RANK"])
        print(os.environ["HOSTNAME"] + " NCCL_SOCKET_IFNAME: " + os.environ["NCCL_SOCKET_IFNAME"])
        print(os.environ["HOSTNAME"] + " NCCL_DEBUG: " + os.environ["NCCL_DEBUG"])
        print(os.environ["HOSTNAME"] + " NCCL_IB_CUDA_SUPPORT: " + os.environ["NCCL_IB_CUDA_SUPPORT"])
        print("Using " + str(config.num_nodes) + " Nodes---------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    else:
        print("Using " + str(config.num_nodes) + " Node----------------------------------------------------------------------")
        print("Using " + str(torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")

    train_config = {'batch_size':config.n_batch, 'num_workers':config.n_workers, 'pin_memory':True}
    ## this should allow us to save a model for every x iterations and it should overwrite
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_top_k=-1, verbose=True)
    train_loader = MoleculeModule(config.max_len, config.train_load, train_config)
    train_loader.setup()#config.debug)
    cachefiles = train_loader.get_cache()

    model = LightningModule(config, train_loader.get_vocab())

    trainer = pl.Trainer(default_root_dir=config.root_dir,
                max_epochs=config.max_epochs,
                accelerator="cuda",
                strategy="ddp",
                num_nodes=config.num_nodes,
                # gpus=config.gpus,
                callbacks=[ModelCheckpointAtEpochEnd(), CheckpointEveryNSteps(config.checkpoint_every)],
                enable_checkpointing=checkpoint_callback,
                accumulate_grad_batches=config.grad_acc,
                num_sanity_val_steps=10,
                val_check_interval=config.eval_every,
                devices=1)
    
                # weights_summary='full')
    try:
        trainer.fit(model, train_loader)
    except Exception as exp:
        print(type(exp))
        print(exp)
        exit()
        # rank_zero_warn('We have caught an error, trying to shut down gracefully')
        # remove_tree(cachefiles)

    if config.debug is True:
        pass
    else:
        exit()
        # rank_zero_warn('Debug mode not found eraseing cache')
        # remove_tree(cachefiles)


if __name__ == '__main__':
    # torch.set_float32_matmul_precision('high')
    main()
