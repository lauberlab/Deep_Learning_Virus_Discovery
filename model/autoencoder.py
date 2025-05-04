import torch
from torch import nn

from .edge_features_A import EdgeFeatures
from .transformer_A import (GraphTransformer, global_pooling, )


class SSEGraphEncoderModel(nn.Module):

    def __init__(self, device, seed: int, x_feature_size: int, edge_feature_size: int, hidden_channels: int,
                 augment_eps: float = 1e-6, num_attn_heads: int = 8,  pooling: str = "sum", num_layers_enc: int = 3,
                 pos_feature_size: int = 16, motif_masking: bool = False, final_mlp: bool = False):
        super(SSEGraphEncoderModel, self).__init__()

        torch.manual_seed(seed)

        self.pooling = pooling
        self.embedding_size = hidden_channels
        self.augment_eps = augment_eps

        self.EdgeFeatures = EdgeFeatures(device=device,
                                         pos_feature_size=pos_feature_size,
                                         edge_feature_size=edge_feature_size,
                                         augment_eps=augment_eps)

        # encoding layers
        self.edge_embedding = nn.Linear(pos_feature_size + 7, edge_feature_size, bias=True)
        self.W_x = nn.Linear(x_feature_size, hidden_channels)

        # init global pooling layer in case we have merged pooling attributes
        if pooling == "merge":
            self.W_p = nn.Linear(hidden_channels * 3, hidden_channels, bias=False)

        else:
            self.register_parameter("W_p", None)

        # dropout layer
        self.D_x = nn.Dropout(0.1)

        # define the encoding layers
        self.transformer_layer = nn.ModuleList()

        # transformer layers
        for _ in range(num_layers_enc):

            self.transformer_layer.append(

                GraphTransformer(out_channels=hidden_channels * 2,
                                 num_attn_heads=num_attn_heads,
                                 edge_feature_size=edge_feature_size)
            )

        # define decoder layers
        self.D_1 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.D_2 = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        # manage whether to add a final classification layer 
        self.use_MLP = final_mlp

        if self.use_MLP:
            self.W_o = nn.Linear(self.out_channels, 1, bias=True)

        else:
            self.W_o = self.register_parameter("classifier", None)

        # additional params
        self.motif_masking = motif_masking
        self.out_channels = self.embedding_size

        # parameter reset
        self.reset_parameters()

        # use Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def reset_parameters(self):
        self.W_e.reset_parameters()
        self.W_x.reset_parameters()

        if self.use_MLP:
            self.W_o.reset_parameters()

        if self.pooling == "merge":
            self.W_p.reset_parameters()

    # forward propagation mechanism
    def forward(self, data):

        """
        B = batch size, N = number of nodes, P = number of edges,
        E = edge channels (param), H = hidden channels (param)
        """

        # extract attributes
        x, p, m, e_idx = data.x, data.p, data.m, data.e                         # x [B, N, 3072]; c [B, N, 3]; m [B, N]

        # x: data augmentation
        if self.training and self.augment_eps > 0.:
            x_noise = self.augment_eps * torch.randn_like(x)
            x = (x + (0.1 * x_noise)) * m.unsqueeze(-1)

        # encode node features 'x'
        h_x = self.W_x(x)                                                       # [B, N, H]

        # encode coordinates 'p' as edge attributes 'h_e'
        h_e = self.EdgeFeatures(p, e_idx, m)                                    # [B, N, E + 7]
        h_e = self.W_e(h_e)                                                     # [B, N, N, E]

        # encode graph embeddings (node-level) utilizing Transformer-architecture
        for transformer in self.transformer_layer:
            h_x = transformer(x_in=h_x, e_in=h_e, m_in=m)                       # [B, N, H]

        # decode graph embeddings for restoration of masked motif domains (node-level);
        # in case pre-training phase is flagged
        if self.motif_masking:
            h_x = self.D_1(h_x).relu()
            h_x = self.D_2(h_x).relu()                                          # [B, N, H]

        # apply global pooling to node-level representations 'h_x' to get graph-level representation 'h_g'
        h_g = global_pooling(self.pooling, h_x, m)

        if self.pooling == "merge":
            h_g = self.W_p(h_g)

        if self.classifier is not None:

            """
            NOTE: using a BCEWithLogitsLoss cost function makes an explicit Sigmoid activation function redundant,
            since it is inherently applied within the loss function itself! Thus, all we need is to return the the class
            logits from the last layer (classifier)..
            """

            h_o = self.D_x(h_g, training=self.training)
            return self.W_o(h_o)

        else:
            return h_g
