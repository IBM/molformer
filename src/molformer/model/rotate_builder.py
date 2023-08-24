from torch.nn import LayerNorm, Module
from src.molformer.model.transformers import VizEncoderLayer, VizEncoder
from src.molformer.model.attention_layer import RotateAttentionLayer 
# from src.molformer.model.attention_weights import LinearWWeight, FullWWeight
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.builders.base import BaseBuilder
from fast_transformers.builders.transformer_builders import BaseTransformerEncoderBuilder
from fast_transformers.builders.attention_builders import AttentionBuilder


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



class RotateEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = RotateEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return RotateAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayer
