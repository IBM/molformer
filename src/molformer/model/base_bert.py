import torch
from torch import nn
import numpy as np
import random
from pubchem_encoder import Encoder
import pytorch_lightning as pl
from pytorch_lightning.utilities import seed

from fast_transformers.builders import TransformerEncoderBuilder
import torch.nn.functional as F
from functools import partial
from apex import optimizers


class LM_Layer(nn.Module):
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


class LightningModule(pl.LightningModule):
    def __init__(self, config, vocab):
        super(LightningModule, self).__init__()

        self.save_hyperparameters(config)
        self.vocabulary = vocab
        # location of cache File
        # Special symbols

        self.debug = config.debug
        self.text_encoder = Encoder(config.max_len)
        # Word embeddings layer
        n_vocab = len(vocab.keys())
        # input embedding stem
        builder = TransformerEncoderBuilder.from_kwargs(  # simple full attention
            n_layers=config.n_layer,
            n_heads=config.n_head,
            query_dimensions=config.n_embd // config.n_head,
            value_dimensions=config.n_embd // config.n_head,
            feed_forward_dimensions=config.n_embd,
            attention_type="fullwweights",
            activation="gelu",
        )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
        self.drop = nn.Dropout(config.d_dropout)
        ## transformer
        self.blocks = builder.get()
        self.lang_model = LM_Layer(config.n_embd, n_vocab)
        self.train_config = config
        # if we are starting from scratch set seeds
        if config.restart_path == "":
            seed.seed_everything(config.seed)

    def lm_forward(self, batch, mask=None, mode="cls"):
        b, t = batch.size()

        token_embeddings = self.tok_emb(
            batch
        )  # each index maps to a (learnable) vector
        if self.pos_emb is not None:
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
        else:
            x = self.drop(token_embeddings)

        if mask is not None:
            x, attention_mask = self.blocks(
                x, length_mask=self.lang_model(mask._mask.sum(-1))
            )

        else:
            x, attention_mask = self.blocks(x)

        if mode == "cls":
            return x[:, 0, :], attention_mask
        elif mode == "max":
            token_embeddings = x
            input_mask_expanded = (
                mask._mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            token_embeddings[
                input_mask_expanded == 0
            ] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            return max_over_time, attention_mask
        elif mode == "avg":
            token_embeddings = x
            input_mask_expanded = (
                mask._mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            )  # sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sum_mask = input_mask_expanded.sum(1)
            return sum_embeddings / sum_mask, attention_mask

    def on_save_checkpoint(self, checkpoint):
        # save RNG states each time the model and states are saved
        out_dict = dict()
        out_dict["torch_state"] = torch.get_rng_state()
        out_dict["cuda_state"] = torch.cuda.get_rng_state()
        if np:
            out_dict["numpy_state"] = np.random.get_state()
        if random:
            out_dict["python_state"] = random.getstate()
        checkpoint["rng"] = out_dict

    def on_load_checkpoint(self, checkpoint):
        # load RNG states each time the model and states are loaded from checkpoint
        rng = checkpoint["rng"]
        for key, value in rng.items():
            if key == "torch_state":
                torch.set_rng_state(value)
            elif key == "cuda_state":
                torch.cuda.set_rng_state(value)
            elif key == "numpy_state":
                np.random.set_state(value)
            elif key == "python_state":
                random.setstate(value)
            else:
                print("unrecognized state")

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
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        if self.pos_emb is not None:
            no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": 0.0,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        betas = (0.9, 0.99)
        learning_rate = self.train_config.lr_start * self.train_config.lr_multiplier
        optimizer = optimizers.FusedLAMB(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        idxl = batch[0]
        targetsl = batch[1]
        # lengthsl = batch[2]

        loss = 0
        loss_tmp = 0
        for chunk in range(len(idxl)):
            idx = idxl[chunk]
            targets = targetsl[chunk]
            len(idx)
            b, t = idx.size()
            # forward the model
            token_embeddings = self.tok_emb(
                idx
            )  # each index maps to a (learnable) vector
            x = self.drop(token_embeddings)
            # masking of the length of the inputs its handled in the Masked language part of the code
            # do not attempt to handle it in the forward of the transformer
            x = self.blocks(x)
            logits = self.lang_model(x)

            # if we are given targets also calculate the loss
            if targets is not None:
                # -- mle loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
                loss_tmp = true_token_lprobs / len(idxl)
            if chunk < len(idxl) - 1:
                loss_tmp.backward()
                loss += loss_tmp.detach()
            else:
                loss += loss_tmp
        self.log("train_loss", loss, on_step=True)
        return {"loss": loss}  # , 'log':tensorboard_log}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([output["loss"] for output in outputs]).mean()
        loss = {"loss": avg_loss.item()}
        self.log("validation_loss", loss["loss"])

    def validation_step(self, batch, batch_idx):
        idxl = batch[0]
        targetsl = batch[1]

        loss = 0
        loss_tmp = 0
        for chunk in range(len(idxl)):
            idx = idxl[chunk]
            targets = targetsl[chunk]
            len(idx)
            b, t = idx.size()
            # forward the model
            token_embeddings = self.tok_emb(
                idx
            )  # each index maps to a (learnable) vector
            x = self.drop(token_embeddings)
            x = self.blocks(x)
            logits = self.lang_model(x)

            # if we are given targets also calculate the loss
            if targets is not None:
                # -- mle loss
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                true_token_lprobs = F.cross_entropy(logits, targets, ignore_index=-100)
                loss_tmp = true_token_lprobs / len(idxl)
            if chunk < len(idxl) - 1:
                loss += loss_tmp.detach()
            else:
                loss += loss_tmp
        self.log("train_loss", loss, on_step=True)
        return {"loss": loss}
