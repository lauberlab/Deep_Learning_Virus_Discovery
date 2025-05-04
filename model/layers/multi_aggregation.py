b"""
adapted by simplifying and changing the aggregation process based on:
(1) PyTorch Geometric MultiAggregation
    ==> https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.aggr.MultiAggregation.html
"""

from typing import Any, Dict, List, Optional
import torch
from torch import Tensor
import torch.nn as nn

from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.nn.aggr import Aggregation, AttentionalAggregation, SumAggregation


class MultiAggregationHandler(Aggregation):

    def __init__(self, aggr_concat: bool, in_channels: int = None,
                 aggrs_kwargs: Optional[List[Dict[str, Any]]] = None):
        super().__init__()

        self.aggr_concat = aggr_concat

        # pick and resolve aggregations
        # ------------------------------------------------------------------------------------------------------------ #

        aggrs = ["sum", "mean", "max"]

        # set default (empty) kwargs for aggregation resolver
        if aggrs_kwargs is None:
            aggrs_kwargs = [{}] * len(aggrs)

        # resolve 'string' aggregations
        if aggrs is not None:
            self.aggrs = nn.ModuleList([
                aggregation_resolver(aggr, **aggr_kwargs) for aggr, aggr_kwargs in zip(aggrs, aggrs_kwargs)
            ])

        else:
            raise ValueError(f"'aggrs' cannot be 'None'.")

        # update out_channels based on aggregation mode
        self.aggr_out_channels = sum([in_channels] * len(self.aggrs)) if aggr_concat else in_channels

    def reset_parameters(self):
        for aggr in self.aggrs:
            aggr.reset_parameters()

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None, dim: int = -2) -> Tensor:

        outs: List[Tensor] = [x] * len(self.aggrs)  # fill with dummy tensors.

        for i, aggr in enumerate(self.aggrs):
            outs[i] = aggr(x, index, ptr, dim_size, dim)

        return self.combine(outs)

    def combine(self, inputs: List[Tensor]) -> Tensor:

        # concatenate aggregators
        if self.aggr_concat:
            return torch.cat(inputs, dim=-1)

        else:
            # stack the aggregators and take the sum of it
            return torch.sum(torch.stack(inputs, dim=0), dim=0)

