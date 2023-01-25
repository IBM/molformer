from math import sqrt

import torch
from torch.nn import Dropout, Module

from fast_transformers.attention_registry import AttentionRegistry, Optional, Float, \
    EventDispatcherInstance
from fast_transformers.events import EventDispatcher, AttentionEvent
from fast_transformers.attention.full_attention import FullAttention 

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

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

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
        #A = self.dropout(torch.softmax(QK, dim=-1))
        A = self.dropout(attention_weights)
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous(), attention_weights.detach()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "fullwweights", FullWWeight,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
