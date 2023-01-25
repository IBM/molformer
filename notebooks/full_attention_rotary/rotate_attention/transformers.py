import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
import torch.nn.functional as F

from fast_transformers.events import EventDispatcher
from fast_transformers.masking import FullMask, LengthMask
from fast_transformers.transformers import TransformerEncoderLayer, TransformerEncoder


class VizEncoderLayer(TransformerEncoderLayer):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements a modification of the Fast
    Transformer layer code where the attention weights of the model are
    returned for vizualization.

    Arguments
    ---------
        These are unchanged from the Fast Transformer encoder layer. For
        further information look there. 
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(VizEncoderLayer, self).__init__(attention, d_model, d_ff=d_ff, dropout=dropout, 
                activation=activation, event_dispatcher=event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        out, attention_mask = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask
        )
        x = x + self.dropout(out)
        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y), attention_mask.detach()


class VizEncoder(TransformerEncoder):

    """A Modifiation of TransformerEncoder where a list of attention weights
        are returned for vizualization purposes. 

    Arguments
    ---------
        These are unchanged from the Fast Transformer encoder. For
        further information look there. 
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(VizEncoder, self).__init__(layers, norm_layer=norm_layer, event_dispatcher=event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        attention_mask_list = []
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for layer in self.layers:
            x, attention_mask = layer(x, attn_mask=attn_mask, length_mask=length_mask)
            attention_mask_list.append(attention_mask)
        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x, attention_mask_list


