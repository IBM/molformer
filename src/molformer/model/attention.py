from math import sqrt
from molformer.model.base_bert import LM_Layer
from molformer.model.rotate_attention.rotate_builder import (
    RotateEncoderBuilder as rotate_builder,
)
from fast_transformers.events import EventDispatcher, AttentionEvent
from fast_transformers.attention_registry import (
    AttentionRegistry,
    Optional,
    Float,
    EventDispatcherInstance,
)
from fast_transformers.builders import TransformerEncoderBuilder
import torch
from torch import nn
from torch.nn import Dropout, Module

from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial


class TestBert(nn.Module):
    def __init__(
        self,
        config,
        vocab,
        model_path=None,
        extend_pos=False,
        rotate=False,
        device="cpu",
    ):
        if model_path is None:
            assert False
        super().__init__()
        torch.load(model_path, map_location=torch.device(device))
        #         config = Namespace(**model["hyper_parameters"])
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
                attention_type="linear",
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
        block_size = block_size

        return tok_emb, pos_emb, blocks, drop, lang_model

    def forward(self, batch, mask=None, mode="cls"):
        b, t = batch.size()

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
            x, attention_mask = self.blocks(x, length_mask=LM_Layer(mask._mask.sum(-1)))

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


class LinearWWeight(Module):
    """
    Slightly modify the fast transformers linear attention to return
    the 'attention weights' for visual analysis
    """

    def __init__(
        self, query_dimensions, feature_map=None, eps=1e-6, event_dispatcher=""
    ):
        super(LinearWWeight, self).__init__()

        self.feature_map = feature_map(query_dimensions)
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(
                ("LinearAttention does not support arbitrary " "attention masks")
            )
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)

        attention = torch.einsum("nlhd, nshd->nlsh", Q, K)
        # try both postive and negative values

        attention_norm = 1 / (torch.einsum("nlsh->nlh", attention + self.eps))
        attention_out = torch.einsum("nlsh, nlh->nlsh", attention, attention_norm)
        attention_out = torch.einsum("nlsh->nhls", attention_out)
        # remove negative numbers with relu

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous(), attention_out.detach()


class FullWWeight(Module):
    """
    Slightly modify the fast transformers Full attention to return
    the 'attention weights' for visual analysis
    """

    def __init__(self, softmax_temp=None, attention_dropout=0.1, event_dispatcher=""):
        super(FullWWeight, self).__init__()

        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1.0 / sqrt(E)

        # Scale the queries instead of applying the softmax temperature to the
        # dot products
        queries = queries * softmax_temp

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        if not key_lengths.all_ones:
            QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        attention_weights = torch.softmax(QK, dim=-1)
        # A = self.dropout(torch.softmax(QK, dim=-1))
        A = self.dropout(attention_weights)
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous(), attention_weights.detach()


def register_attention(attention_type="linear"):
    # Register the attention implementation so that it becomes available in our
    # builders
    if attention_type == "linear":
        weights = FullWWeight

    else:
        weights = LinearWWeight

    AttentionRegistry.register(
        f"{attention_type}wweights",
        weights,
        [
            ("softmax_temp", Optional(Float)),
            ("attention_dropout", Optional(Float, 0.1)),
            ("event_dispatcher", Optional(EventDispatcherInstance, "")),
        ],
    )
