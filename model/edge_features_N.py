from torch import nn
import torch
import torch.nn.functional as F
from .transformer_N import gather_nodes
import numpy as np


class LearnableRelativePE(nn.Module):

    def __init__(self, embedding_dim, scale: bool = True):
        super(LearnableRelativePE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_frequencies = embedding_dim // 2

        # learnable frequencies per one sinusoidal component
        self.t_frq = nn.Parameter(
            torch.exp(torch.randn(self.num_frequencies)) * -(np.log(10000.0) / self.embedding_dim)
        )

        # optional learnable scaling vector for stabilization
        if scale:
            self.t_s = nn.Parameter(torch.tensor(1.0))

        else:
            self.register_parameter("t_s", None)

    def forward(self, edge_index):

        _, num_nodes, _ = edge_index.size()

        ii = torch.arange(num_nodes, dtype=torch.float32, device=edge_index.device).view(1, -1, 1)
        d = (edge_index.float() - ii).unsqueeze(-1)

        FR = self.t_frq * self.t_s if self.t_s is not None else self.t_frq
        angles = d * FR.view(1, 1, 1, -1)

        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)                            # [B, L, K, H]


class EdgeFeatures(nn.Module):

    def __init__(self, device, num_pos_features: int = 16, num_edge_features: int = 16, distance_threshold: float = 7.5,
                 augment_eps: float = 1e-6, neighbour_fraction: float = 0.8,
                 num_min_neighbours: int = 15):
        super(EdgeFeatures, self).__init__()

        # self.distance_threshold = distance_threshold
        self.neighbour_fraction = neighbour_fraction
        # self.num_min_neighbours = num_min_neighbours

        # positional encoding layer
        self.LRPE = LearnableRelativePE(num_pos_features)

        # init edge embedding layer
        self.edge_embedding = nn.Linear(num_pos_features + 7, num_edge_features, bias=True)

        self.device = device
        self.augment_eps = augment_eps                                                                        # [B, L, K]

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))

        xyz = signs * magnitudes

        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        return F.normalize(Q, dim=-1)

    def _orientations(self, X, E_idx, eps=1e-6):

        # pairwise differences of consecutive nodes along the sequence; shape: [B, L-1, 3]
        dX = X[:, 1:, :] - X[:, :-1, :]

        # normalized unit vectors
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]                      # all but the last two unit vectors
        u_1 = U[:, 1:-1, :]                     # all but the first and last unit vectors

        # Backbone Normals
        # defines a plane and a direction that captures local geometry and curvature between consecutive nodes
        # n_2 represents a vector orthogonal to the plane formed by the involved nodes; normalization ensures this is
        # a vector unit --> local reference frame
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)

        # Relative Orientations
        # this local frame encodes how each segment of the sequence is oriented relative to the global 3D space

        # bisector vectors between u2 and u1; shape: [B, L-2, 3]
        # capturing the "average direction" between u2 and u1 --> local orientation
        o_1 = F.normalize(u_2 - u_1, dim=-1)

        # local orientation matrices (local coordinate system) constructed from bisectors o1, normals n_2 and
        # the cross product of the two former (orthogonal to both o1 and n_2)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), dim=2)

        # padding to ensure dimensionality
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)                                                   # [B, L, 9]

        # Neighbour Information
        # orientation matrices of a node's neighbours are gathered, describing how each neighbour
        # is oriented in 3D space
        O_neighbours = gather_nodes(O, E_idx)                                                 # [B, L, K, 9]

        # neighbour coordinates are gathered to compute relative displacements and rotations
        X_neighbours = gather_nodes(X, E_idx)                                                 # [B, L, K, 3]

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])                                                      # [B, L, 3, 3]
        O_neighbours = O_neighbours.view(list(O_neighbours.shape[:3]) + [3, 3])                     # [B, L, L, 3, 3]

        # Neighbour displacements
        dX = X_neighbours - X.unsqueeze(-2)                                                         # [B, L, K, 3]

        # .. transformed into local frame using O; normalizing ensure to focus on directional relationships rather
        # than the scale of distances
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)                             # [B, L, K, 3]
        dU = F.normalize(dU, dim=-1)

        # rotations between neighbours and a node
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbours)                            # [B, L, K, 3, 3]

        # Quaternion representation ---> 4 values instead of a 3x3 matrix
        Q = self._quaternions(R)                                                                    # [B, L, K, 4]

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)                                                     # [B, L, K, 7]

        return O_features

    def forward(self, p, e_idx, mask):

        # compute distance matrix and get edge indices
        # E_idx = self._edge_indices(C, mask)

        # C: data augmentation
        if self.training and self.augment_eps > 0.:
            p_noise = self.augment_eps * torch.randn_like(p)
            p = p + 0.1 * p_noise * mask.unsqueeze(-1)

        # compute local orientations for each node including spatial information regarding their neighbours
        q_feats = self._orientations(p, e_idx)                                                      # [B, L, K, 4]

        # compute positional encodings
        e_pe = self.LRPE(e_idx)

        e_cat = torch.cat((e_pe, q_feats), -1)

        # apply linear transformation
        h_e = self.edge_embedding(e_cat)                                                                  # [B, L, N]

        return F.normalize(h_e)  # , E_idx  # edge_indices.transpose(0, 1)                                # 2x [B, L, N]


