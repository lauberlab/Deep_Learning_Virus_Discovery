from typing import List

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import OptTensor


def gather_nodes(node_features, E_idx):
    """
    node_features: tensor of shape [B, L, F] (e.g., coordinates or orientation matrices)
    E_idx: tensor of shape [B, L, K], neighbour indices

    returns: neighbour features of shape [B, L, K, F]

    """

    B, L, F = node_features.shape
    _, _, K = E_idx.shape

    # inflate and expand E_idx to match feature dimensions for indexing
    E_idx_exp = E_idx.unsqueeze(-1).expand(-1, -1, -1, F)  # [B, L, K, F]

    # gather features of neighboring nodes
    neighbour_features = torch.gather(node_features.unsqueeze(1).expand(-1, L, -1, F),
                                      dim=2,
                                      index=E_idx_exp)

    # masking invalid neighbours, in case there are padded values
    neighbour_features[E_idx == -1] = 0

    return neighbour_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)

    return h_nn


def mean_pooling(h_x, mask):
    if mask is not None:

        return torch.sum(h_x, dim=1) / torch.sum(mask, dim=1, keepdim=True) + 1e-9

    else:
        return torch.mean(h_x, dim=1)


def sum_pooling(h_x):
    return torch.sum(h_x, dim=1)


def max_pooling(h_x, mask):
    if mask is not None:

        # assigning large negative values to masked positions to avoid maxing those out
        h_x = h_x + (1 - mask.unsqueeze(-1).float()) * -1e9

    h_pool, _ = torch.max(h_x, dim=1)

    return h_pool


def global_pooling(pooling, h_xu, mask):

    if pooling == "mean":
        h_pool = mean_pooling(h_xu, mask)

    elif pooling == "sum":
        h_pool = sum_pooling(h_xu)

    elif pooling == "max":
        h_pool = max_pooling(h_xu, mask)

    elif pooling == "merge":

        h_mean = mean_pooling(h_xu, mask)
        h_max = max_pooling(h_xu, mask)
        h_sum = sum_pooling(h_xu)

        return torch.cat([h_mean, h_max, h_sum], dim=1)

    else:
        raise ValueError(f"Unsupported pooling type: {pooling}")

    return h_pool


class NeighborAttention(nn.Module):
    def __init__(self, in_channels, out_channels, attn_heads=4):
        super(NeighborAttention, self).__init__()

        self.attn_heads = attn_heads
        self.out_channels = out_channels

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(out_channels, out_channels, bias=False)
        self.W_K = nn.Linear(in_channels, out_channels, bias=False)
        self.W_V = nn.Linear(in_channels, out_channels, bias=False)
        self.W_O = nn.Linear(out_channels * 3, out_channels, bias=False)

        self._reset_parameters()

    def _reset_parameters(self):
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_V.reset_parameters()
        self.W_O.reset_parameters()

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):

        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf).cuda())
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend

        return attend

    def _aggregation(self, h_v: Tensor, attention: Tensor, shape: List):

        # incorporate neighbour importance into aggregation utilizing the attention weights on h_V
        attn_sum = attention.sum(dim=3, keepdim=True) + 1e-8
        weighted_v = (attention.unsqueeze(-1) * h_v.transpose(2, 3))
        weighted_sum = weighted_v.sum(dim=3)

        # derive the mean aggregation over all neighbours (h_v)
        aggr_mean = weighted_sum / attn_sum
        aggr_mean = aggr_mean.view(shape)

        # re-view the sum aggregation
        aggr_sum = weighted_sum.view(shape)

        # deduce max aggregation of all neighbours
        aggr_max, _ = weighted_v.max(dim=3)
        aggr_max = aggr_max.view(shape)

        aggr_cat = torch.cat([aggr_mean, aggr_sum, aggr_max], dim=-1)

        return self.W_O(aggr_cat)

    def forward(self, h_X, h_E, mask_attn):

        # Queries, Keys, Values
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.attn_heads

        d = int(self.out_channels / n_heads)

        h_q = self.W_Q(h_X).view([n_batch, n_nodes, 1, n_heads, 1, d])
        h_k = self.W_K(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        h_v = self.W_V(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        # attention scores
        attn_score = torch.matmul(h_q, h_k).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2, -1)
        attn_score = attn_score / np.sqrt(d)

        if mask_attn is not None:
            mask = mask_attn.unsqueeze(2).expand(-1, -1, self.attn_heads, -1)
            attn_weights = self._masked_softmax(attn_score, mask)

        else:
            attn_weights = F.softmax(attn_score, -1)

        # aggregation & update
        return self._aggregation(h_v, attn_weights, [n_batch, n_nodes, self.out_channels])


class GraphTransformer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, num_attn_heads: int):
        super(GraphTransformer, self).__init__()

        self.T_x = NeighborAttention(in_channels=in_channels,
                                     out_channels=out_channels,
                                     attn_heads=num_attn_heads,
                                     )

        self.O_x = nn.Linear(out_channels, out_channels)
        self.norm_Ox = nn.LayerNorm(out_channels)
        self.norm_Tx = nn.LayerNorm(out_channels)

        # parameter reset
        self.reset_parameters()

    def reset_parameters(self):
        self.O_x.reset_parameters()

    def forward(self, x_in: Tensor, e_in: Tensor, mask_pad: OptTensor = None, mask_attn: OptTensor = None):

        # multi head attention with multi aggregation output
        x_attn = self.T_x(x_in, e_in, mask_attn)

        # dropout + normalization
        h_x = F.dropout(self.norm_Tx(x_attn), 0.1, training=self.training)

        # linear transformation + residues
        h_x = self.O_x(h_x) + x_attn

        # second dropout + normalization
        h_x = F.dropout(self.norm_Ox(h_x), 0.1, training=self.training)

        # masking out padded sections of h_x
        if mask_pad is not None:
            h_x = mask_pad.unsqueeze(-1) * h_x

        return h_x
