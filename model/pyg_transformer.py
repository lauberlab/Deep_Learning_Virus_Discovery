b"""
transferred DGL GraphTransformer implementation to PyTorch Geometric based on:
(1) (1) 'A Generalization of Transformer Networks to Graphs' (Dwivedi et al. 2021)
    ==> https://github.com/graphdeeplearning/graphtransformer/blob/main/layers/graph_transformer_edge_layer.py
"""

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.typing import Adj, OptTensor
from .layers.neighbor_attention import MultiHeadAttentionLayer


class GraphTransformer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_attn_heads: int, aggr_concat: bool = False):
        super(GraphTransformer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_attn_heads = num_attn_heads

        self.multi_attention = MultiHeadAttentionLayer(in_channels=in_channels,
                                                       out_channels=out_channels // num_attn_heads,
                                                       attn_heads=num_attn_heads,
                                                       aggr_concat=aggr_concat)

        self.O_x = nn.Linear(out_channels, out_channels)
        self.norm_Ox = nn.LayerNorm(out_channels)

        # parameter reset
        self.reset_parameters()

    def reset_parameters(self):
        self.O_x.reset_parameters()

    def forward(self, x: Tensor, e: Tensor, edge_index: Adj):

        # initial residual connection
        x0 = x

        # multi head attention with multi aggregation output
        x_attn = self.multi_attention(x, e, edge_index)
        # x = x_attn.view(-1, self.out_channels)

        # dropout
        x = F.dropout(x_attn, 0.1, training=self.training)

        # linear transformation + batch normalization + residues
        x = self.O_x(x) + x0
        x = self.norm_Ox(x)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, attn_heads={self.num_attn_heads})')