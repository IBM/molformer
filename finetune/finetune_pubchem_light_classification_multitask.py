import time
import torch
import torch.nn.functional as F
from torch import nn
import args
import os
import numpy as np
import random
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from tokenizer.tokenizer import MolTranBertTokenizer
from fast_transformers.masking import LengthMask as LM
from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
from apex import optimizers
import subprocess
from argparse import ArgumentParser, Namespace
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from torch.utils.data import DataLoader
from rdkit import Chem
#from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
#from pytorch_lightning.plugins.sharded_plugin import DDPShardedPlugin

def normalize_smiles(smi, canonical, isomeric):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized

class MultitaskModel(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, tokenizer):
        super(MultitaskModel, self).__init__()

        self.config = config
        self.hparams = config
        self.mode = config.mode
        self.save_hyperparameters(config)
        self.tokenizer=tokenizer
        #location of cache File
        # Special symbols
        self.min_loss = {
            self.hparams.dataset_name + "min_valid_loss": torch.finfo(torch.float32).max,
            self.hparams.dataset_name + "min_epoch": 0,
        }

        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), config.n_embd
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd//config.n_head,
            value_dimensions=config.n_embd//config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type='linear',
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
        #########################################
        # protein_emb_dim, smiles_embed_dim, dims=dims, dropout=0.2):
        #########################################

        self.fcs = []  # nn.ModuleList()
        # self.dropout = nn.Dropout(p=dropout)
        self.loss = torch.nn.BCELoss()

        self.net = self.Net(
            config.n_embd, self.hparams.num_tasks, dims=config.dims, dropout=config.dropout,
        )

        #logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))


    class Net(nn.Module):
        dims = [150, 50, 50, 2]


        def __init__(self, smiles_embed_dim, num_tasks, dims=dims, dropout=0.2):
            super().__init__()
            self.desc_skip_connection = True
            self.fcs = []  # nn.ModuleList()
            print('dropout is {}'.format(dropout))

            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, num_tasks) #classif

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)

            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb

            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)
            #z = F.log_softmax(z) #classif
            #z = self.layers(smiles_emb)
            return F.sigmoid(z)

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

    def get_loss(self, smiles_emb, measures, mask):
        z_pred = self.net.forward(smiles_emb).squeeze()
        #measures = measures.long()
        z_pred = z_pred * mask
        #print('z_pred:', z_pred.shape)
        #print('measures:', measures.shape)
        return self.loss(z_pred, measures), z_pred, measures, mask

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
        if self.hparams.measure_name == 'r2':
            betas = (0.9, 0.999)
        else:
            betas = (0.9, 0.99)
        print('betas are {}'.format(betas))
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        idx = batch[0]
        mask = batch[1]
        targets = batch[2]
        target_masks = batch[3]

        loss = 0
        loss_tmp = 0
        b, t = idx.size()
        token_embeddings =self.tok_emb(idx)
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        loss, pred, actual, masks = self.get_loss(loss_input, targets, target_masks)
        #logits = self.lang_model(x)

        self.log('train_loss', loss, on_step=True)

        logs = {"train_loss": loss}

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx, dataset_idx):
        idx = val_batch[0]
        mask = val_batch[1]
        targets = val_batch[2]
        target_masks = val_batch[3]


        loss = 0
        loss_tmp = 0
        b, t = idx.size()
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))
        token_embeddings = x
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
        loss, pred, actual, target_masks = self.get_loss(loss_input, targets, target_masks)
        self.log('train_loss', loss, on_step=True)
        return {
            "val_loss": loss,
            "pred": pred.detach(),
            "actual": actual.detach(),
            "mask":target_masks.detach(),
            "dataset_idx": dataset_idx,
        }
    def validation_epoch_end(self, outputs):
        # results_by_dataset = self.split_results_by_dataset(outputs)
        tensorboard_logs = {}
        for dataset_idx, batch_outputs in enumerate(outputs):
            dataset = self.hparams.dataset_names[dataset_idx]

            preds = torch.cat([x["pred"] for x in batch_outputs])
            actuals = torch.cat([x["actual"] for x in batch_outputs])
            masks = torch.cat([x["mask"] for x in batch_outputs])


            #val_loss, pred, actual, target_masks = self.get_loss(preds, actuals, masks)
            masks = masks > 0

            #val_loss = self.loss(preds, actuals)

            #val_loss = self.loss(preds, actuals)

            actuals_cpu = actuals.detach()
            preds_cpu = preds.detach()

            num_tasks = preds.size()[1]
            aucs = []
            vals = []
            for i in range(num_tasks):
                actuals_task = torch.masked_select(actuals_cpu[:,i], masks[:,i])
                preds_task = torch.masked_select(preds_cpu[:,i], masks[:,i])
                val_loss_task = self.loss(preds_task, actuals_task)
                auc=roc_auc_score(actuals_task.cpu().numpy(),preds_task.cpu().numpy())
                vals.append(val_loss_task)
                aucs.append(auc)
            average_auc = torch.mean(torch.tensor(aucs))
            val_loss = torch.mean(torch.tensor(vals))
            print(dataset, ' loss:', val_loss.item())
            print(dataset, ' auc:', average_auc.item())
            tensorboard_logs.update(
                {
                    # dataset + "_avg_val_loss": avg_loss,
                    self.hparams.dataset_name + "_" + dataset + "_loss": val_loss,
                    self.hparams.dataset_name + "_" + dataset + "_auc": average_auc,
                }
            )

        if (
            tensorboard_logs[self.hparams.dataset_name + "_valid_loss"]
            < self.min_loss[self.hparams.dataset_name + "min_valid_loss"]
        ):
            self.min_loss[self.hparams.dataset_name + "min_valid_loss"] = tensorboard_logs[
                self.hparams.dataset_name + "_valid_loss"
            ]
            self.min_loss[self.hparams.dataset_name + "min_test_loss"] = tensorboard_logs[
                self.hparams.dataset_name + "_test_loss"
            ]
            self.min_loss[self.hparams.dataset_name + "max_valid_auc"] = tensorboard_logs[
                self.hparams.dataset_name + "_valid_auc"
            ]
            self.min_loss[self.hparams.dataset_name + "max_test_auc"] = tensorboard_logs[
                self.hparams.dataset_name + "_test_auc"
            ]
            self.min_loss[self.hparams.dataset_name + "best_epoch"] = self.current_epoch


            tensorboard_logs[self.hparams.dataset_name + "min_valid_loss"] = tensorboard_logs[
                self.hparams.dataset_name + "_valid_loss"
            ]
            tensorboard_logs[self.hparams.dataset_name + "min_test_loss"] = tensorboard_logs[
                self.hparams.dataset_name + "_test_loss"
            ]
            tensorboard_logs[self.hparams.dataset_name + "max_valid_auc"] = tensorboard_logs[
                self.hparams.dataset_name + "_valid_auc"
            ]
            tensorboard_logs[self.hparams.dataset_name + "max_test_auc"] = tensorboard_logs[
                self.hparams.dataset_name + "_test_auc"
            ]



        self.logger.log_metrics(tensorboard_logs, self.global_step)

        self.log(self.hparams.dataset_name + "_" + dataset + "_loss", val_loss)

        print("Validation: Current Epoch", self.current_epoch)
        append_to_file(
            os.path.join(self.hparams.results_dir, "results_" + self.hparams.dataset_name+ "_"+ os.environ["LSB_JOBID"]+".csv"),
            #os.path.join(self.hparams.results_dir, "results_" + self.hparams.dataset_name+ ".csv"),
            f"{self.hparams.dataset_name}, {self.current_epoch},"
            + f"{tensorboard_logs[self.hparams.dataset_name + '_valid_loss']},"
            + f"{tensorboard_logs[self.hparams.dataset_name + '_test_loss']},"
            + f"{tensorboard_logs[self.hparams.dataset_name + '_valid_auc']},"
            + f"{tensorboard_logs[self.hparams.dataset_name + '_test_auc']},"

            + str(float(self.min_loss[self.hparams.dataset_name + "min_valid_loss"]))
            + ","
            + str(float(self.min_loss[self.hparams.dataset_name + "min_test_loss"]))
            + ","
            + str(float(self.min_loss[self.hparams.dataset_name + "max_valid_auc"]))
            + ","
            + str(float(self.min_loss[self.hparams.dataset_name + "max_test_auc"]))
            + ","
            + str(int(self.min_loss[self.hparams.dataset_name + "best_epoch"]))


        )

        # return {"avg_val_loss": avg_loss, "log": tensorboard_logs}
        return {"avg_val_loss": val_loss}


def get_dataset(data_root, filename,  dataset_len,  measure_names):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = MultitaskEmbeddingDataset(df,  measure_names)
    return dataset

class MultitaskEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, df,  measure_names):
        self.measure_names = measure_names
        df['canonical_smiles'] = df['smiles'].apply(lambda smi: normalize_smiles(smi, canonical=True, isomeric=False))
        df_good = df.dropna(subset=['canonical_smiles'])  # TODO - Check why some rows are na

        len_new = len(df_good)
        print('Dropped {} invalid smiles'.format(len(df) - len_new))
        self.df = df_good

        self.df = self.df.reset_index(drop=True)

    def __getitem__(self, index):

        canonical_smiles = self.df.loc[index, 'canonical_smiles']
        measures = self.df.loc[index, self.measure_names].to_numpy()
        mask = [0.0 if np.isnan(x) else 1.0 for x in measures]
        measures = [0.0 if np.isnan(x) else x for x in measures]
        return canonical_smiles, measures, mask

    def __len__(self):
        return len(self.df)

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(PropertyPredictionDataModule, self).__init__()
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.smiles_emb_size = hparams.n_embd
        self.tokenizer = MolTranBertTokenizer('bert_vocab.txt')
        self.dataset_name = hparams.dataset_name

    def get_split_dataset_filename(dataset_name, split):
        return split + ".csv"

    def prepare_data(self):
        print("Inside prepare_dataset")
        train_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "train"
        )

        valid_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "valid"
        )

        test_filename = PropertyPredictionDataModule.get_split_dataset_filename(
            self.dataset_name, "test"
        )

        train_ds = get_dataset(
            self.hparams.data_root,
            train_filename,
            self.hparams.train_dataset_length,
            measure_names=self.hparams.measure_names,
        )

        val_ds = get_dataset(
            self.hparams.data_root,
            valid_filename,
            self.hparams.eval_dataset_length,
            measure_names=self.hparams.measure_names,
        )

        test_ds = get_dataset(
            self.hparams.data_root,
            test_filename,
            self.hparams.eval_dataset_length,
            measure_names=self.hparams.measure_names,
        )

        self.train_ds = train_ds
        self.val_ds = [val_ds] + [test_ds]

        # print(
        #     f"Train dataset size: {len(self.train_ds)}, val: {len(self.val_ds1), len(self.val_ds2)}, test: {len(self.test_ds)}"
        # )

    def collate(self, batch):
        tokens = self.tokenizer.batch_encode_plus([ smile[0] for smile in batch], padding=True, add_special_tokens=True)
        return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor([smile[1] for smile in batch], dtype=torch.float32), torch.tensor([smile[2] for smile in batch]))

    def val_dataloader(self):
        return [
            DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                shuffle=False,
                collate_fn=self.collate,
            )
            for ds in self.val_ds
        ]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
        )



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
                #filename = f"{self.prefix}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class ModelCheckpointAtEpochEnd(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics['epoch'] = trainer.current_epoch
        if trainer.disable_validation:
            trainer.checkpoint_callback.on_validation_end(trainer, pl_module)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")

def main():
    margs = args.parse_args()
    print("Using " + str(
        torch.cuda.device_count()) + " GPUs---------------------------------------------------------------------")
    pos_emb_type = 'rot'
    if margs.dataset_name =='tox21':
        margs.measure_names =  ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                                'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    elif margs.dataset_name =='clintox':
        margs.measure_names = ['FDA_APPROVED', 'CT_TOX']
    elif margs.dataset_name =='muv':
        margs.measure_names = [
            'MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
            'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
            'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859'
        ]
    elif margs.dataset_name =='sider':
        margs.measure_names = [
            'Hepatobiliary disorders', 'Metabolism and nutrition disorders',
            'Product issues', 'Eye disorders', 'Investigations',
            'Musculoskeletal and connective tissue disorders',
            'Gastrointestinal disorders', 'Social circumstances',
            'Immune system disorders', 'Reproductive system and breast disorders',
            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
            'General disorders and administration site conditions',
            'Endocrine disorders', 'Surgical and medical procedures',
            'Vascular disorders', 'Blood and lymphatic system disorders',
            'Skin and subcutaneous tissue disorders',
            'Congenital, familial and genetic disorders', 'Infections and infestations',
            'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders',
            'Renal and urinary disorders',
            'Pregnancy, puerperium and perinatal conditions',
            'Ear and labyrinth disorders', 'Cardiac disorders',
            'Nervous system disorders', 'Injury, poisoning and procedural complications'
        ]
    elif margs.dataset_name =='hiv':
        margs.measure_names = [
            "HIV_active"
        ]


    margs.num_tasks = len(margs.measure_names)


    run_name_fields = [
        margs.dataset_name,

        pos_emb_type,
        margs.fold,
        margs.mode,
        "lr",
        margs.lr_start,
        "batch",
        margs.batch_size,
        "drop",
        margs.dropout,
    ]

    run_name = "_".join(map(str, run_name_fields))
    try:
        jobid = os.environ["LSB_JOBID"]
    except:
        print("JOBID env variable not set using 0000")
        jobid = '0000'

    print(run_name)
    datamodule = PropertyPredictionDataModule(margs)
    margs.dataset_names = "valid test".split()
    margs.run_name = run_name

    checkpoints_folder = margs.checkpoints_folder
    checkpoint_root = os.path.join(checkpoints_folder, margs.dataset_name)
    margs.checkpoint_root = checkpoint_root
    os.makedirs(checkpoints_folder, exist_ok=True)
    checkpoint_dir = os.path.join(checkpoint_root, "models_"+jobid)
    results_dir = os.path.join(checkpoint_root, "results")
    margs.results_dir = results_dir
    margs.checkpoint_dir = checkpoint_dir
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoints_folder, margs.measure_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(period=1, save_last=True, dirpath=checkpoint_dir, filename='checkpoint', verbose=True)

    print(margs)

    logger = TensorBoardLogger(
        save_dir=checkpoint_root,
        #version=run_name,
        name="lightning_logs",
        default_hp_metric=False,
    )

    tokenizer = MolTranBertTokenizer('bert_vocab.txt')
    seed.seed_everything(margs.seed)

    if margs.seed_path == '':
        print("# training from scratch")
        model = MultitaskModel(margs, tokenizer)
    else:
        print("# loaded pre-trained model from {args.seed_path}")
        model = MultitaskModel(margs, tokenizer).load_from_checkpoint(margs.seed_path, strict=False, config=margs, tokenizer=tokenizer, vocab=len(tokenizer.vocab))


    last_checkpoint_file = os.path.join(checkpoint_dir, "last.ckpt")
    resume_from_checkpoint = None
    if os.path.isfile(last_checkpoint_file):
        print(f"resuming training from : {last_checkpoint_file}")
        resume_from_checkpoint = last_checkpoint_file
    else:
        print(f"training from scratch")

    trainer = pl.Trainer(
        max_epochs=margs.max_epochs,
        default_root_dir=checkpoint_root,
        gpus=1,
        logger=logger,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_callback=checkpoint_callback,
        num_sanity_val_steps=0,
    )

    tic = time.perf_counter()
    trainer.fit(model, datamodule)
    toc = time.perf_counter()
    print('Time was {}'.format(toc - tic))


if __name__ == '__main__':
    main()
