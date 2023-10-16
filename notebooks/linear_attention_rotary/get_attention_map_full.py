from argparse import Namespace
import torch.nn.functional as F

from tokenizer.tokenizer import MolTranBertTokenizer
from utils import normalize_smiles
import torch
import shutil
from torch import nn
import args
import os
import getpass
from datasets import load_dataset, concatenate_datasets, load_from_disk

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import FullMask, LengthMask as LM
from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
import fast_transformers.attention.linear_attention
import rotate_attention.linear_attention

from fast_transformers.feature_maps import Favor, GeneralizedRandomFeatures
from functools import partial

from torch.utils.data import DataLoader


class TestBert(nn.Module):
    def __init__(
        self, vocab, model_path=None, extend_pos=False, rotate=False, device="cpu"
    ):
        if model_path == None:
            assert False
        super().__init__()
        model = torch.load(model_path, map_location=torch.device(device))
        config = Namespace(**model["hyper_parameters"])
        config.rotate = rotate
        (
            self.tok_emb,
            self.pos_emb,
            self.blocks,
            self.drop,
            self.lang_model,
        ) = self.create_model(config, vocab)
        """if self.pos_emb != None:
            pos_emb = nn.Parameter(model['state_dict']['pos_emb'])

        if extend_pos is True and self.pos_emb != None:
            pos_extend = nn.Parameter(torch.zeros(1, 1000, config.n_embd)).to(pos_emb.device)
            self.pos_emb = nn.Parameter(torch.cat([pos_emb.data, pos_extend.data], dim=1))"""

        self.drop = nn.Dropout(config.d_dropout)

    def create_model(self, config, vocab):

        n_vocab, d_emb = len(vocab.keys()), config.n_embd
        block_size = 250
        if config.rotate:
            builder = rotate_builder.from_kwargs(
                n_layers=config.n_layer,
                n_heads=config.n_head,
                query_dimensions=config.n_embd // config.n_head,
                value_dimensions=config.n_embd // config.n_head,
                feed_forward_dimensions=config.n_embd,
                attention_type="linearwweights",
                feature_map=partial(
                    GeneralizedRandomFeatures,
                    n_dims=config.num_feats,
                    deterministic_eval=True,
                ),
                activation="gelu",
            )
            pos_emb = None
        else:
            builder = TransformerEncoderBuilder.from_kwargs(
                n_layers=config.n_layer,
                n_heads=config.n_head,
                query_dimensions=config.n_embd // config.n_head,
                value_dimensions=config.n_embd // config.n_head,
                feed_forward_dimensions=config.n_embd,
                attention_type="linearwweights",
                feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats, deterministic_eval=True),
                activation="gelu",
            )
            pos_emb = nn.Parameter(torch.zeros(1, block_size, config.n_embd))
        tok_emb = nn.Embedding(n_vocab, config.n_embd)
        drop = nn.Dropout(config.d_dropout)

        blocks = builder.get()
        lang_model = lm_layer(config.n_embd, n_vocab)
        train_config = config
        block_size = block_size

        return tok_emb, pos_emb, blocks, drop, lang_model

    def forward(self, batch, mask=None, mode="cls"):
        b, t = batch.size()

        # forward the GPT model
        token_embeddings = self.tok_emb(
            batch
        )  # each index maps to a (learnable) vector
        if self.pos_emb != None:
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
        else:
            x = self.drop(token_embeddings)

        if mask != None:
            x, attention_mask = self.blocks(x, length_mask=LM(mask._mask.sum(-1)))

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


def get_database(config):
    pubchem_path = {
        "train": "/dccstor/trustedgen/data/pubchem/CID-SMILES-CANONICAL.smi"
    }
    if "CANONICAL" in pubchem_path:
        pubchem_script = "./pubchem_canon_script.py"
    else:
        pubchem_script = "./pubchem_script.py"
        dataset_dict = load_dataset(
            pubchem_script,
            data_files=pubchem_path,
            cache_dir=os.path.join(
                "/tmp", getpass.getuser(), "pubchem_{}".format(config.chunk_num)
            ),
            split="train",
        )
    train_config = {
        "batch_size": config.n_batch,
        "shuffle": False,
        "num_workers": config.n_workers,
        "pin_memory": True,
    }
    # loader =  DataLoader(dataset_dict, **train_config)
    print(dataset_dict.cache_files)
    cache_files = []
    for cache in dataset_dict.cache_files:
        tmp = "/".join(cache["filename"].split("/")[:4])
        print(tmp)
        cache_files.append(tmp)

    print("dataset length {}".format(len(dataset_dict)))
    if 50000 * config.chunk_num > len(dataset_dict):
        index_end = 0
        loader = None
    elif 50000 + 50000 * config.chunk_num > len(dataset_dict):
        index_end = 50000 + 50000 * config.chunk_num - len(dataset_dict)
        index = [i + (50000 * config.chunk_num) for i in range(index_end)]
        loader = torch.utils.data.Subset(dataset_dict, index)
        loader = DataLoader(loader, **train_config)
    else:
        index_end = 50000
        index = [i + (50000 * config.chunk_num) for i in range(index_end)]
        loader = torch.utils.data.Subset(dataset_dict, index)
        loader = DataLoader(loader, **train_config)
    # index= [i+(50000*config.chunk_num) for i in range(index_end)])
    return loader, cache_files


def get_bert(config, tokenizer):
    bert_model = TestBert(
        tokenizer.vocab, config.seed_path, rotate=config.rotate, device=config.device
    ).to(config.device)
    tmp_model = torch.load(config.seed_path, map_location=torch.device(config.device))["state_dict"]
    bert_model.load_state_dict(tmp_model, strict=True)
    bert_model = bert_model.eval()
    return bert_model


def remove_tree(cachefiles):
    if type(cachefiles) == type([]):
        cachefiles = list(set(cachefiles))
        for cache in cachefiles:
            shutil.rmtree(cache)
    else:
        shutil.rmtree(cachefiles)


def get_tokens_from_ids(input_ids, tokenizer):
    tokens = []

    for idx_lst in input_ids:
        seq = []
        for idx in idx_lst:
            seq.append(tokenizer.ids_to_tokens[idx])
        tokens.append(seq)
    return tokens


def get_full_attention(molecule):
    config = args.parse_args()
    model_path = config.seed_path
    device = config.device
    batch_size = config.batch_size
    canonical = config.canonical
    mode = config.mode
    mask = config.mask

    loader = None
    tokenizer = MolTranBertTokenizer("bert_vocab.txt")
    bert_model = get_bert(config, tokenizer)

    batch_total = 0
    if loader is not None:

        for batch_number, mols in enumerate(loader):
            batch_to_save = []
            with torch.no_grad():

                # print(batch_number)
                if config.canonical is True:
                    output = [
                        normalize_smiles(smiles, canonical=True, isomeric=False)
                        for smiles in mols["text"]
                        if smiles is not None
                    ]
                else:
                    output = mols["text"]
                batch_ids = tokenizer.batch_encode_plus(
                    output,
                    padding=True,
                    add_special_tokens=True,
                    return_attention_mask=True,
                    return_length=True,
                )

                if config.mask is True:
                    att_mask = FullMask(
                        torch.tensor(batch_ids["attention_mask"], dtype=bool).to(
                            device
                        ),
                        device=device,
                    )
                else:
                    att_mask = FullMask(
                        torch.ones(
                            torch.tensor(batch_ids["input_ids"]).size(), dtype=bool
                        ).to(device),
                        device=device,
                    )

                embeddings, attention_mask = bert_model(
                    torch.tensor(batch_ids["input_ids"]).to(device),
                    att_mask,
                    mode=config.mode,
                )

            for number, mol in enumerate(output):
                batch_to_save.append((embeddings[number].cpu().numpy(), mol))

            # if len(batch_to_save) >= 500:
            batch_name = "batch_num_{}.pth".format(
                batch_number + (50000 * config.chunk_num)
            )
            chunk_name = "chunk_num_{}".format(config.chunk_num)
            if batch_number % 250 == 0:
                print(batch_name)
            torch.save(
                batch_to_save[0],
                os.path.join("./embedding_dump_deterministic", chunk_name, batch_name),
            )

    else:
        with torch.no_grad():

            if config.canonical is True:
                output = [normalize_smiles(molecule, canonical=True, isomeric=False)]
            else:
                output = molecule

            batch_ids = tokenizer.batch_encode_plus(
                [output],
                padding=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_length=True,
            )

            raw_tokens = get_tokens_from_ids(batch_ids["input_ids"], tokenizer)[0]

            if config.mask is True:
                att_mask = FullMask(
                    torch.tensor(batch_ids["attention_mask"], dtype=bool).to(device),
                    device=device,
                )
            else:
                att_mask = FullMask(
                    torch.ones(
                        torch.tensor(batch_ids["input_ids"]).size(), dtype=bool
                    ).to(device),
                    device=device,
                )

            embeddings, attention_mask = bert_model(
                torch.tensor(batch_ids["input_ids"]).to(device),
                att_mask,
                mode=config.mode,
            )
            return attention_mask, raw_tokens

    if loader != None:
        remove_tree(cache_files)


if __name__ == "__main__":
    attentions = get_full_attention()
