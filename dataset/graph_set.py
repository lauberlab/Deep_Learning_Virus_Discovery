from typing import List
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import traceback
import logging
import numpy as np
import gc
import psutil
import os

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class BatchData:

    def __init__(self, x, p, m, e, y):

        self.x = x
        self.p = p
        self.m = m
        self.e = e
        self.y = y

    def detach(self):

        if isinstance(self.x, torch.Tensor):
            self.x = self.x.detach()

        if isinstance(self.p, torch.Tensor):
            self.p = self.p.detach()

        if isinstance(self.m, torch.Tensor):
            self.m = self.m.detach()

        if isinstance(self.e, torch.Tensor):
            self.e = self.e.detach()

        if isinstance(self.y, torch.Tensor):
            self.y = self.y.detach()

        return self

    def to_numpy(self):

        if isinstance(self.x, torch.Tensor):
            self.x = self.x.cpu().numpy()

        if isinstance(self.p, torch.Tensor):
            self.p = self.p.cpu().numpy()

        if isinstance(self.m, torch.Tensor):
            self.m = self.m.cpu().numpy()

        if isinstance(self.e, torch.Tensor):
            self.e = self.e.cpu().numpy()

        if isinstance(self.y, torch.Tensor):
            self.y = self.y.cpu().numpy()

        return self


class GraphDataSet(Dataset):

    def __init__(self, data_dict: dict):
        super().__init__()

        self.x = torch.stack(data_dict["feats"])
        self.p = torch.stack(data_dict["coordinates"])
        self.m = torch.stack(data_dict["masks"])
        self.e = torch.stack(data_dict["edge_indices"])
        self.y = torch.stack(data_dict["labels"])

        assert self.x.shape[0] == self.p.shape[0] == self.y.shape[0] == self.m.shape[0] == self.e.shape[0]
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.x[index], self.p[index], self.m[index], self.e[index], self.y[index],


class GraphSubset(Dataset):

    def __init__(self, original_dataset, indices):
        super().__init__()

        self.x = original_dataset.x[indices]
        self.p = original_dataset.p[indices]
        self.m = original_dataset.m[indices]
        self.e = original_dataset.e[indices]
        self.y = original_dataset.y[indices]

        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.x[idx], self.p[idx], self.m[idx], self.e[idx], self.y[idx]


def collate_fn(batch):

    x_batch = torch.stack([b[0] for b in batch])  # stack inputs
    p_batch = torch.stack([b[1] for b in batch])  # stack coordinates
    m_batch = torch.stack([b[2] for b in batch])  # stack masks
    e_batch = torch.stack([b[3] for b in batch])  # stack edge indices
    y_batch = torch.stack([b[4] for b in batch])  # stack labels

    return BatchData(x=x_batch, p=p_batch, m=m_batch, e=e_batch, y=y_batch)


def vector_normalization(vector: np.ndarray):

    norm = np.linalg.norm(vector)
    norm_vector = vector if norm == 0 else vector / norm

    return norm_vector


def embedding_aggregation(embedding: np.ndarray):
    """
    we aggregate multiple embedding reductions like summation, averaging, min/max pooling

    :param:  embedding of shape (N, L, 1024) where N is the number of nodes and L the number of amino acid
             embeddings per node n
    :return: stacked numpy matrix of shape (3, N, 1024) unified over L by applying different reduction methods
    """

    sum_embs, avg_embs, max_embs = [], [], []

    for emb in embedding:

        emb = emb.astype(np.float32)

        sum_embs.append(vector_normalization(np.sum(emb, axis=0)))
        avg_embs.append(vector_normalization(np.mean(emb, axis=0)))
        max_embs.append(vector_normalization(np.max(emb, axis=0)))

    embeddings: List[Tensor] = [torch.tensor(sum_embs),
                                torch.tensor(avg_embs),
                                torch.tensor(max_embs)]

    return torch.cat(embeddings, dim=-1)


def compute_edge_indices(coordinates: Tensor, mask: Tensor, num_min_neighbours: int,
                         f: float,  T: float = 7.5, eps=1E-9):

    # coordinates 'C'
    C_expanded = coordinates.unsqueeze(1)                                       # [L, 1, 3]
    C_transposed = coordinates.unsqueeze(0)                                     # [L, 3, 1]

    pair_diff = torch.abs(C_expanded - C_transposed)                            # [L, L, 3]
    pair_dist = torch.sqrt(torch.sum(pair_diff**2, dim=2) + eps)                # [L, L] distance matrix (Euclidean)

    # create 2D mask
    mask_2D = mask.unsqueeze(0) * mask.unsqueeze(1)                             # [L, L]

    # apply mask
    pair_dist[mask_2D == 0] = float('inf')

    # deploy distance threshold 'T' and mask out values exceeding set threshold
    mask_dist = pair_dist <= T                                                  # [L, L] binary mask (True, False)
    val_dist = pair_dist.clone()
    val_dist[~mask_dist] = float('inf')                                         # masked distance matrix (cloned)

    # sorting
    sort_dist, sort_idx = torch.sort(val_dist, dim=-1)                          # [L, L]
    max_neighbours = mask_dist.sum(dim=1).max().item()
    num_neighbours_to_select = max(num_min_neighbours, int(max_neighbours * f))

    # truncate to the maximum number of neighbours
    # this behaviour is similar to Top-K algorithm
    return sort_idx[:, :num_neighbours_to_select]                               # [L, K] where K == num_neighbours


def compute_edge_indices_backbone(coordinates: torch.Tensor, mask: torch.Tensor, num_min_neighbours: int,
                                  f: float, T: float = 7.5, eps=1E-9):

    L = coordinates.shape[0]
    device = coordinates.device

    # 1. backbone connectivity --> connect each subsequent data point in mask (SSE order) ruling out padding entries
    # ---------------------------------------------------------------------------------------------------------------- #
    # binary matrix
    backbone_edges = torch.zeros((L, L), dtype=torch.bool, device=device)

    for i in range(L - 1):

        if mask[i] and mask[i + 1]:
            backbone_edges[i, i + 1] = True
            backbone_edges[i + 1, i] = True

    # 2. Distance-based connectivity --> based on coordinates entries
    # ---------------------------------------------------------------------------------------------------------------- #
    C_expanded = coordinates.unsqueeze(1)  # [L, 1, 3]
    C_transposed = coordinates.unsqueeze(0)  # [L, 3, 1]

    pair_diff = torch.abs(C_expanded - C_transposed)  # [L, L, 3]
    pair_dist = torch.sqrt(torch.sum(pair_diff**2, dim=2) + eps)  # [L, L], Euclidean distance

    # apply mask vector to identify padded regions
    mask_2D = mask.unsqueeze(0) * mask.unsqueeze(1)  # [L, L]
    pair_dist[mask_2D == 0] = float('inf')

    # apply distance threshold to assign edge to nodes meeting this criterion only
    mask_dist = pair_dist <= T  # [L, L]
    val_dist = pair_dist.clone()
    val_dist[~mask_dist] = float('inf')

    # neighbour selection --> imitating torch.topk
    sort_dist, sort_idx = torch.sort(val_dist, dim=-1)  # [L, L]
    max_neighbours = mask_dist.sum(dim=1).max().item()
    num_neighbours_to_select = max(num_min_neighbours, int(max_neighbours * f))

    # combine backbone and distance-based edges
    final_adj_matrix = backbone_edges.clone()  # [L, L]

    # iterate over all nodes L
    for i in range(L):

        if backbone_edges[i].any():  # If node has direct (backbone) neighbours, skip distance-based for those
            continue

        # extract all neighbours for current node
        neighbours = sort_idx[i, :num_neighbours_to_select].tolist()

        # add edge for all neighbours of current node
        # only if they are valid neighbours (not padded) and don't point to themselves
        for neighbour in neighbours:

            if mask[neighbour] and neighbour != i:
                final_adj_matrix[i, neighbour] = True

    # convert adjacency matrix to edge index [L,K]
    edge_indices = torch.zeros((L, num_neighbours_to_select), dtype=torch.long, device=device)

    for i in range(L):

        neighbors = torch.where(final_adj_matrix[i])[0]
        num_neighbors = neighbors.shape[0]

        if num_neighbors > 0:
            edge_indices[i, :min(num_neighbours_to_select, num_neighbors)] = neighbors[:min(num_neighbours_to_select,
                                                                                            num_neighbors)]
        else:
            edge_indices[i, :] = -1                                                  # fill with -1 if no neighbors.

    return edge_indices


def transform_data(data, name: str, num_min_neighbours: int, neighbour_fraction: float, references: list = None,
                   distance_threshold: float = 7.5, rdrp_only: bool = False, motif_mask: bool = False):

    # define outputs
    data_dict = {
        "feats": [], "coordinates": [], "labels": [], "masks": [], "edge_indices": [], "motif_masks": [],
    }

    reference_dict = {
        "feats": [], "coordinates": [], "labels": [], "masks": [], "edge_indices": [], "motif_masks": [],
    }

    # extract data contents & load what's needed
    # x = embeddings | p = positions  | y = labels | h = headers | m = binary mask indices
    x, p, y, h = data['x'], data['p'], data['y'], data['h']

    m = data["m"] if motif_mask else None

    # when 'rdrp_only' flagged --> consider only samples from the positive class
    pos_idx = np.where(y == 1)[0]

    if rdrp_only:
        x, p, y, h = x[pos_idx], p[pos_idx], y[pos_idx], h[pos_idx]
        m = m[pos_idx] if m is not None else m

    # number of data samples
    num_samples = x.shape[0]

    # determine max number nodes 'L' for padding operation
    L = max([x[i].shape[0] for i in range(x.shape[0] - 1)]) if num_samples > 1 else x.shape[-1]
    print(f"padding size: {L}")

    # expect_dim_x = x[0].shape[-1]

    for idx in range(0, num_samples):

        if idx > 0 and idx % 10000 == 0:
            logging.info(f" ----- {idx + 1} samples processed..")

        # if idx == 10000:
        #    break

        try:
            # 0 - get header and convert it to datatype in accordance with torch tensor types
            # -------------------------------------------------------------------------------------------------------- #
            header = h[idx]

            if isinstance(type(header), np.ndarray):
                header = header.tostring().decode('utf-8')

            # 1 - node features and embeddings
            # -------------------------------------------------------------------------------------------------------- #
            # get node data from 'x'

            """
            emb data should have size (N, ) where N is the number of nodes in the current graph sample;
            each n of N represents an SSE composed of L amino acids; thus, when zooming into the embedding on 
            node-level each n-th embedding has size (L, 1024) 

            this corresponds to x having a size of (N, L, 1024) per graph
            """

            # reduce embeddings
            embedding = x[idx]

            # for amino acid graphs
            if len(embedding.shape) == 2:
                embedding = np.expand_dims(embedding, axis=1)

            embeddings = embedding_aggregation(embedding)
            padding = torch.full((L - embeddings.size(0), embeddings.size(1)), 0.)
            embedding_data = torch.cat([embeddings, padding])  # padded embeddings

            # 2 - positional data (coordinates)
            # -------------------------------------------------------------------------------------------------------- #
            pos_data = torch.tensor(np.array(p[idx]).astype(float), dtype=torch.float32)
            padding = torch.full((L - pos_data.size(0), pos_data.size(1)), 0.)
            coordinates_data = torch.cat([pos_data, padding])  # padded coordinates

            # assert pos_data.size(0) == embeddings.size(0)
            if pos_data.size(0) != embeddings.size(0):
                print(f"dim mismatch: {pos_data.size(0)} != {embeddings.size(0)}!")
                continue

            if coordinates_data.size(0) != embedding_data.size(0):
                print(f"pad mismatch: {coordinates_data.size(0)} != {embedding_data.size(0)}!")
                continue

            # 3 - add motif masking
            # -------------------------------------------------------------------------------------------------------- #
            motif_mask_data = m

            if m is not None:
                motif_mask = torch.tensor(m[idx].reshape(-1, 1).astype(int), dtype=torch.int)
                padding = torch.full((L - motif_mask.size(0), motif_mask.size(1)), -1)
                motif_mask_data = torch.cat([motif_mask, padding])

            # 4 - generate mask for embeddings and positions
            # -------------------------------------------------------------------------------------------------------- #
            mask_1 = torch.full((pos_data.size(0),), 1.)
            mask_0 = torch.full((L - pos_data.size(0),), 0.)
            mask_data = torch.cat([mask_1, mask_0]).bool()

            # 5 - generate mask for embeddings and positions
            # -------------------------------------------------------------------------------------------------------- #
            edge_indices = compute_edge_indices(coordinates=coordinates_data,
                                                mask=mask_data,
                                                num_min_neighbours=num_min_neighbours,
                                                f=neighbour_fraction, T=distance_threshold,)

            # 6 - get labels and transform to one hot vectors if flagged
            # -------------------------------------------------------------------------------------------------------- #
            label_data = torch.tensor(int(y[idx]))

            if references is not None and header in references:
                reference_dict["feats"].append(embedding_data)
                reference_dict["coordinates"].append(coordinates_data)
                reference_dict["masks"].append(mask_data)
                reference_dict["edge_indices"].append(edge_indices)
                reference_dict["labels"].append(label_data)
                reference_dict["motif_masks"].append(motif_mask_data)

            else:
                data_dict["feats"].append(embedding_data)
                data_dict["coordinates"].append(coordinates_data)
                data_dict["masks"].append(mask_data)
                data_dict["edge_indices"].append(edge_indices)
                data_dict["labels"].append(label_data)
                data_dict["motif_masks"].append(motif_mask_data)

        except Exception as e:
            logging.error(f"@sample {h[idx]} #{idx+1} ==> {e}")
            traceback.print_exc()

    # 7 - free RAM
    # ---------------------------------------------------------------------------------------------------------------- #
    mem_collected = get_memory_usage_mb()

    del x, p, y, h, m
    del label_data, edge_indices, mask_data, motif_mask_data, coordinates_data, pos_data, embedding_data
    del embeddings, padding
    gc.collect()

    mem_freed = mem_collected - get_memory_usage_mb()

    logging.info(f"ran garbage collection --> freed {mem_freed:.2f} MB.")
    logging.info(f"{name} set contains {len(data_dict['feats'])} samples")

    return data_dict, reference_dict, L










