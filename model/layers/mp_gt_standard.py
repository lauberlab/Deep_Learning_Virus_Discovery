import math
from typing import Optional, Union

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax
from .multi_aggregation import MultiAggregationHandler


class MultiHeadAttentionLayer(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, attn_heads: int, dropout: float = 0.,
                 attn_root: bool = False, aggr_concat: bool = False,
                 **kwargs):
        super(MultiHeadAttentionLayer, self).__init__(node_dim=0, aggr=None, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.attn_root = attn_root
        self.aggr_concat = aggr_concat

        # aggregation handling within message passing layer
        self.aggregations = MultiAggregationHandler(aggr_concat=aggr_concat, in_channels=out_channels)

        # in case aggregators are concatenated; propagate through linear layer to unify shapes and let the model
        # learn what aggregators might have the biggest effect
        if aggr_concat:
            aggr_out_channels = self.aggregations.aggr_out_channels
            self.linear_concat = nn.Linear(aggr_out_channels, out_channels)

        else:
            self.linear_concat = self.register_parameter("linear_concat", None)

        # mirror input channels
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        # attention layers
        self.query = nn.Linear(in_channels[1], out_channels * attn_heads)
        self.key = nn.Linear(in_channels[0], out_channels * attn_heads)
        self.value = nn.Linear(in_channels[0], out_channels * attn_heads)

        # register edge attribute projection layer
        self.edge_proj = nn.Linear(in_channels[0], out_channels * attn_heads, bias=False)

        # register skip-connect layer for attention roots (unfiltered inputs 'x')
        if attn_root:
            self.skip = nn.Linear(in_channels[1], self.out_channels * attn_heads, bias=True)

        else:
            self.skip = self.register_parameter("skip", None)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        self.edge_proj.reset_parameters()

        if self.aggr_concat:
            self.linear_concat.reset_parameters()

        if self.attn_root:
            self.skip.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr: OptTensor = None):

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        q_x = self.query(x[1]).view(-1, self.attn_heads, self.out_channels)
        k_x = self.key(x[0]).view(-1, self.attn_heads, self.out_channels)
        v_x = self.value(x[0]).view(-1, self.attn_heads, self.out_channels)

        # standard propagation
        x_out = self.propagate(edge_index, q=q_x, k=k_x, v=v_x, edge_attr=edge_attr, size=None)

        # in case we have concatenated aggregations propagate through linear layer first
        if self.aggr_concat:
            x_out = self.linear_concat(x_out)

        # reduce to dimension of 'out_channels' by concatenating each attention head
        x_out = x_out.view(-1, self.attn_heads * self.out_channels)

        # add transformed root node features to the aggregated output
        if self.attn_root:
            x_r = self.skip(x[1])
            x_out = x_out + x_r

        return x_out

    def message(self, q_i: Tensor, k_j: Tensor, v_j: Tensor, edge_attr: OptTensor, index: Tensor,
                ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        if edge_attr is not None and self.edge_proj is not None:
            edge_attr = self.edge_proj(edge_attr).view(-1, self.attn_heads, self.out_channels)

        # scaling & modifying attention score by multiplying with edge attributes
        kq_score = (q_i * k_j) / math.sqrt(self.out_channels)
        attention_score = kq_score + edge_attr if edge_attr is not None else kq_score

        # softmax
        attention = softmax(attention_score.sum(dim=-1), index, ptr, size_i)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        # compute message
        message = v_j + edge_attr if edge_attr is not None else v_j
        message = message * attention.view(-1, self.attn_heads, 1)

        return message

    def aggregate(self, message: Tensor, index: Tensor, ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:

        # utilizing multi aggregation of received messages at target nodes
        a = self.aggregations(message, index, ptr=ptr, dim_size=dim_size, dim=self.node_dim)

        return a

    def update(self, aggr):

        return super(MultiHeadAttentionLayer, self).update(aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, attn_heads={self.attn_heads})')