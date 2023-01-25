import torch
from torch.nn import Module

from fast_transformers.attention_registry import AttentionRegistry, Optional, Callable, Int, \
    EventDispatcherInstance
from fast_transformers.attention.linear_attention import LinearAttention 
from fast_transformers.events import EventDispatcher


class LinearWWeight(Module):
    """
        Slightly modify the fast transformers linear attention to return
        the 'attention weights' for visual analysis
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6, 
                 event_dispatcher=""):
        super(LinearWWeight, self).__init__()

        self.feature_map = (feature_map(query_dimensions))
        self.eps = eps
        self.event_dispatcher=EventDispatcher.get(event_dispatcher)
    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        attention = torch.einsum('nlhd, nshd->nlsh', Q, K)
        #try both postive and negative values

        
        attention_norm = 1/(torch.einsum('nlsh->nlh', attention+self.eps))
        attention_out = torch.einsum('nlsh, nlh->nlsh', attention, attention_norm)
        attention_out = torch.einsum('nlsh->nhls', attention_out)
        #remove negative numbers with relu

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous(), attention_out.detach()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "linearwweights", LinearWWeight,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
