from molformer.model.base_bert import LM_Layer
from molformer.model.attention_modules.rotate_builder import (
    RotateEncoderBuilder as rotate_builder,
)
from fast_transformers.builders import TransformerEncoderBuilder
import torch
from torch import nn

from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
from fast_transformers.masking import LengthMask
from argparse import Namespace


class TestBert(nn.Module):
    def __init__(
        self,
        vocab,
        model_path=None,
        rotate=False,
        device="cpu",
    ):
        if model_path is None:
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
                feature_map=partial(
                    GeneralizedRandomFeatures,
                    n_dims=config.num_feats,
                    deterministic_eval=True,
                ),
                activation="gelu",
            )
            pos_emb = nn.Parameter(torch.zeros(1, block_size, config.n_embd))
        tok_emb = nn.Embedding(n_vocab, config.n_embd)
        drop = nn.Dropout(config.d_dropout)

        blocks = builder.get()
        lang_model = LM_Layer(config.n_embd, n_vocab)

        return tok_emb, pos_emb, blocks, drop, lang_model

    def forward(self, batch, mask=None, mode="cls"):
        _batch, t = batch.size()

        # forward the GPT model
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
                x, length_mask=LengthMask(mask._mask.sum(-1))
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
