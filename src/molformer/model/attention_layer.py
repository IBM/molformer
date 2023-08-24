"""

The rotate attention layer performs all the query key value projections and
output projections leaving the implementation of the attention to the inner
attention module.

"""

from torch.nn import Linear, Module
import torch
from fast_transformers.attention import AttentionLayer
from fast_transformers.events import EventDispatcher, QKVEvent


class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = 0 
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            #if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None,:, None, :]
            self.sin_cached = emb.sin()[None,:, None, :]
            #else:
            #    cos_return = self.cos_cached[..., :seq_len]
            #    sin_return = self.sin_cached[..., :seq_len]
            #    return cos_return, sin_return
                
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in earlier torch versions

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)



class RotateAttentionLayer(AttentionLayer):
    """Rotate attention layer inherits from fast_transformer attention layer. 
        The only thing added is an Embedding encoding, for more information
        on the attention layer see the fast_transformers code
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(RotateAttentionLayer, self).__init__(attention,d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)

        self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using Rotation Embedding')

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """
        Using the same frame work as the fast_Transformers attention layer
        but injecting rotary information to the queries and the keys
        after the keys and queries are projected. 
        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        cos, sin = self.rotaryemb(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        values = self.value_projection(values).view(N, S, H, -1)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))


        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)
