import argparse
import glob
import torch
import shutil
import rdkit
from torch import nn
import args
import os
import numpy as np
import random
import getpass
from datasets import load_dataset, concatenate_datasets, load_from_disk
from pubchem_encoder import Encoder
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from pytorch_lightning.callbacks import LearningRateMonitor

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import LengthMask as LM
from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
#from fast_trans_code.builders import TransformerEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import Favor,GeneralizedRandomFeatures
import torch.nn.functional as F
from functools import partial
from apex import optimizers

from torch.utils.data import DataLoader
import subprocess

class LightningModule(pl.LightningModule):

    def __init__(self, config, vocab):
        super(LightningModule, self).__init__()

        self.save_hyperparameters(config)
        self.vocabulary = vocab
        #location of cache File
        # Special symbols

        self.debug = config.debug
        self.text_encoder = Encoder(config.max_len)
        # Word embeddings layer
        n_vocab, d_emb = len(vocab.keys()), config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
            #attention_type='full',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(config.n_embd, n_vocab)
        self.train_config = config
        #if we are starting from scratch set seeds
        if config.restart_path == "":
            seed.seed_everything(config.seed)




    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def on_save_checkpoint(self, checkpoint):
        #save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict['torch_state']=torch.get_rng_state()
        out_dict['cuda_state']=torch.cuda.get_rng_state()
        if np:
            out_dict['numpy_state']=np.random.get_state()
        if random:
            out_dict['python_state']=random.getstate()
        checkpoint['rng'] = out_dict

    def on_load_checkpoint(self, checkpoint):
        #load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint['rng']
        for key, value in rng.items():
            if key =='torch_state':
                torch.set_rng_state(value)
            elif key =='cuda_state':
                torch.cuda.set_rng_state(value)
            elif key =='numpy_state':
                np.random.set_state(value)
            elif key =='python_state':
                random.setstate(value)
            else:
                print('unrecognized state')

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)


        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        betas = (0.9, 0.99)
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        idxl =     batch[0]
        targetsl = batch[1]
        #lengthsl = batch[2]

        loss = 0
        loss_tmp = 0
        for chunk in range(len(idxl)):
            idx = idxl[chunk]
            targets = targetsl[chunk]
            b_element_size = len(idx)
            b, t = idx.size()
            # forward the model
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            x = self.drop(token_embeddings)
            #masking of the length of the inputs its handled in the Masked language part of the code
            #do not attempt to handle it in the forward of the transformer
            x = self.blocks(x)
            logits = self.lang_model(x)

            # if we are given targets also calculate the loss
            if targets is not None:
                # -- mle loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
                loss_tmp = true_token_lprobs/len(idxl)
            if chunk < len(idxl)-1:
                loss_tmp.backward()
                loss += loss_tmp.detach()
            else:
                loss += loss_tmp
        self.log('train_loss', loss, on_step=True)
        return {'loss':loss}#, 'log':tensorboard_log}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.tensor([output['loss'] for output in outputs]).mean()
        loss = {'loss': avg_loss.item()}
        self.log('validation_loss', loss['loss'])
    def validation_step(self, batch, batch_idx):
        idxl =     batch[0]
        targetsl = batch[1]

        loss = 0
        loss_tmp = 0
        for chunk in range(len(idxl)):
            idx = idxl[chunk]
            targets = targetsl[chunk]
            b_element_size = len(idx)
            b, t = idx.size()
            # forward the model
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            x = self.drop(token_embeddings)
            x = self.blocks(x)
            logits = self.lang_model(x)

            # if we are given targets also calculate the loss
            if targets is not None:
                # -- mle loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
                loss_tmp = true_token_lprobs/len(idxl)
            if chunk < len(idxl)-1:
                loss += loss_tmp.detach()
            else:
                loss += loss_tmp
        self.log('train_loss', loss, on_step=True)
        return {'loss':loss}

class MoleculeModule(pl.LightningDataModule):
    def __init__(self,  max_len, data_path, train_args):
        super().__init__()
        self.data_path = data_path
        self.train_args = train_args  # dict with keys {'batch_size', 'shuffle', 'num_workers', 'pin_memory'}
        print(train_args)
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
        pubchem_path = {'train':'../data/pubchem/CID-SMILES-CANONICAL.smi'}
        if 'CANONICAL' in pubchem_path:
            pubchem_script = './pubchem_canon_script.py'
        else:
            pubchem_script = './pubchem_script.py'
        zinc_path = '../data/ZINC'
        if 'ZINC' in self.data_path or 'zinc' in self.data_path:
            zinc_files = [f for f in glob.glob(os.path.join(zinc_path,'*.smi'))]
            for zfile in zinc_files:
                print(zfile)
            self.data_path = {'train': zinc_files}
            dataset_dict = load_dataset('./zinc_script.py', data_files=self.data_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'zinc'),split='train')

        elif 'pubchem' in self.data_path:
            dataset_dict =  load_dataset(pubchem_script, data_files=pubchem_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'pubchem'), split='train')
        elif 'both' in self.data_path or 'Both' in self.data_path or 'BOTH' in self.data_path:
            dataset_dict_pubchem =  load_dataset(pubchem_script, data_files=pubchem_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'pubchem'),split='train')
            zinc_files = [f for f in glob.glob(os.path.join(zinc_path,'*.smi'))]
            for zfile in zinc_files:
                print(zfile)
            self.data_path = {'train': zinc_files}
            dataset_dict_zinc =  load_dataset('./zinc_script.py', data_files=self.data_path, cache_dir=os.path.join('/tmp',getpass.getuser(), 'zinc'),split='train')
            dataset_dict = concatenate_datasets([dataset_dict_zinc, dataset_dict_pubchem])
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

    def val_dataloader(self):
        pass
    def test_dataloader(self):
        pass
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
    # os.environ['NCCL_SOCKET_IFNAME'] = "^lo,docker,virbr,vmnet,vboxnet,wl,ww,ppp,bond"

    # ifname = os.environ.get('NCCL_SOCKET_IFNAME', None)
    # if ifname is None:
    #     os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker0'

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



def main():
    fix_infiniband()

    config = args.parse_args()
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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(period=1, save_top_k=-1, verbose=True)
    train_loader = MoleculeModule(config.max_len, config.train_load, train_config)
    train_loader.setup()#config.debug)
    cachefiles = train_loader.get_cache()
    model = LightningModule(config, train_loader.get_vocab())

    trainer = pl.Trainer(default_root_dir=config.root_dir,
                max_epochs=config.max_epochs,
                accelerator=config.accelerator,
                num_nodes=config.num_nodes,
                gpus=config.gpus,
                callbacks=[ModelCheckpointAtEpochEnd(), CheckpointEveryNSteps(config.checkpoint_every)],
                checkpoint_callback=checkpoint_callback,
                resume_from_checkpoint=config.restart_path if config.restart_path != "" else None,
                accumulate_grad_batches=config.grad_acc,
                num_sanity_val_steps=10,
                val_check_interval=config.eval_every,
                weights_summary='full')
    try:
        trainer.fit(model, train_loader)
    except Exception as exp:
        print(type(exp))
        print(exp)
        rank_zero_warn('We have caught an error, trying to shut down gracefully')
        remove_tree(cachefiles)

    if config.debug is True:
        pass
    else:
        rank_zero_warn('Debug mode not found eraseing cache')
        remove_tree(cachefiles)


if __name__ == '__main__':

    main()
