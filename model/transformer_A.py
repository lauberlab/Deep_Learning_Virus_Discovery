import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import OptTensor


def sum_pooling(h_x):
    return h_x.sum(dim=1)


def mean_pooling(h_x, m_exp):

    if m_exp is not None:

        return sum_pooling(h_x) / m_exp.sum(dim=1).clamp(min=1e-8)

    else:
        return h_x.mean(dim=1)


def max_pooling(h_x, m_exp):

    if m_exp is not None:

        # assigning large negative values to masked positions to avoid maxing those out
        h_x = h_x.masked_fill(m_exp == 0, float("-inf"))

    h_pool, _ = torch.max(h_x, dim=1)

    return h_pool


def global_pooling(pooling, h_x, mask):

    if mask is not None:
        mask_exp = mask.unsqueeze(-1).float()                           # [B, N, 1]
        h_x *= mask_exp

    if pooling == "mean":
        h_pool = mean_pooling(h_x, mask)

    elif pooling == "sum":
        h_pool = sum_pooling(h_x)

    elif pooling == "max":
        h_pool = max_pooling(h_x, mask)

    elif pooling == "merge":

        h_mean = mean_pooling(h_x, mask)
        h_max = max_pooling(h_x, mask)
        h_sum = sum_pooling(h_x)

        return torch.cat([h_mean, h_max, h_sum], dim=1)

    else:
        raise ValueError(f"Unsupported pooling type: {pooling}")

    return h_pool


class EdgeAwareMultiHeadAttention(nn.Module):
    def __init__(self, hidden_channels: int, edge_feature_size: int, attn_heads: int = 4):
        super(EdgeAwareMultiHeadAttention, self).__init__()

        self.attn_heads = attn_heads
        self.out_channels = hidden_channels / 2

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(hidden_channels, self.out_channels, bias=False)
        self.W_K = nn.Linear(edge_feature_size, self.out_channels, bias=False)
        self.W_V = nn.Linear(edge_feature_size, self.out_channels, bias=False)

        # output transformation after multi-aggregation (mean + sum + max)
        self.W_R = nn.Linear(3 * hidden_channels, self.out_channels, bias=False)

        # parameter reset (after each initialization)
        self._reset_parameters()

    def _reset_parameters(self):
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_V.reset_parameters()
        self.W_R.reset_parameters()

    def forward(self, h_x, h_e, h_m):

        # Queries, Keys, Values
        B, N = h_e.shape[:2]
        H = self.attn_heads
        D = int(self.out_channels / H)

        """
        linear projections for queries (from node features), keys/values (from edge features)
        adapted based on https://github.com/WeiLab-Biology/DeepProSite/
        """
        h_q = self.W_Q(h_x).view([B, N, H, D]).transpose(1, 2)                                      # [B, H, N, D]
        h_k = self.W_K(h_e).view([B, N, N, H, D]).permute(0, 3, 1, 2, 4)                            # [B, H, N, N, D]
        h_v = self.W_V(h_e).view([B, N, N, H, D]).permute(0, 3, 1, 2, 4)                            # [B, H, N, N, D]

        # attention scores
        hq_exp = h_q.unsqueeze(3)                                                                   # [B, H, N, 1, D]
        attn_scores = torch.sum(hq_exp * h_k, dim=-1) / np.sqrt(D)                                  # [B, H, N, N]

        # apply masking, partially converting zero-padded into -inf values; which may end in propagating NaN values;
        # may cause problems when logits are completely masked (edge cases, should not occur due to filtering)
        if h_m is not None:
            hm_exp = h_m.unsqueeze(1).unsqueeze(-1)                                                 # [B, 1, 1, N]
            attn_scores = attn_scores.masked_fill(hm_exp == 0, float('-inf'))

        # attention normalizing using softmax function
        attn_scores = F.softmax(attn_scores, dim=1)                                                 # [B, H, N, N]

        # weighted attention scores
        attn_weighted = attn_scores.unsqueeze(-1) * h_v                                             # [B, H, N, N, D]

        # multi-aggregation: sum, mean, max
        aggr_sum = torch.sum(attn_weighted, dim=3)                                                  # [B, H, N, D]

        attn_sum = attn_scores.sum(dim=3, keepdim=True) + 1e-8
        aggr_mean = aggr_sum / attn_sum                                                             # [B, H, N, D]

        aggr_max, _ = attn_weighted.max(dim=3)                                                      # [B, H, N, D]

        aggr_multi = torch.cat([aggr_mean, attn_sum, aggr_max], dim=-1)                             # [B, H, N, 3D]
        aggr_out = aggr_multi.permute(0, 2, 1, 3).contiguous().view(B, N, 3 * H * D)                # [B, H, N, 3 * hidden_channels]

        # attentive reduction
        return self.W_R(aggr_out)                                                                   # [B, N, out_channels]


class GraphTransformer(nn.Module):

    def __init__(self, edge_feature_size: int, out_channels: int, num_attn_heads: int):
        super(GraphTransformer, self).__init__()

        self.T_x = EdgeAwareMultiHeadAttention(hidden_channels=out_channels,
                                               attn_heads=num_attn_heads,
                                               edge_feature_size=edge_feature_size,)

        self.O_x = nn.Linear(out_channels, out_channels)
        self.norm_Ox = nn.LayerNorm(out_channels)
        self.norm_Tx = nn.LayerNorm(out_channels)

        # parameter reset
        self.reset_parameters()

    def reset_parameters(self):
        self.O_x.reset_parameters()

    def forward(self, x_in: Tensor, e_in: Tensor, m_in: OptTensor = None):

        # multi head attention with multi aggregation output
        x_attn = self.T_x(x_in, e_in, m_in)

        # dropout + normalization
        h_x = F.dropout(self.norm_Tx(x_attn), 0.1, training=self.training)

        # linear transformation + residues
        h_x = self.O_x(h_x) + x_attn

        # second dropout + normalization
        h_x = F.dropout(self.norm_Ox(h_x), 0.1, training=self.training)

        # masking out padded sections of h_x
        if m_in is not None:
            h_x = m_in.unsqueeze(-1) * h_x

        return h_x
