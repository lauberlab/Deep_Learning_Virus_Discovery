import torch
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class FixedAnchorTripletMiner(BaseMiner):

    def __init__(self, distance, anchor_emb: torch.Tensor, margin: float = 0.2, batch_size: int = 32,
                 type_of_triplets="hard", **kwargs):
        super(FixedAnchorTripletMiner, self).__init__(**kwargs)

        self.margin = margin
        self.distance = distance
        self.batch_size = batch_size
        self.anchor_emb = anchor_emb
        self.type_of_triplets = type_of_triplets
        self.add_to_recordable_attributes(list_of_names=["margin"], is_stat=False)
        self.add_to_recordable_attributes(
            list_of_names=["avg_triplet_margin", "pos_pair_dist", "neg_pair_dist"],
            is_stat=True,
        )

    def mine(self, embeddings, labels, ref_emb=None, ref_label=None):

        # compute distances
        mat = self.distance(embeddings, self.anchor_emb)

        # get all triplets
        a, p, n = lmu.get_all_triplet_indices(labels, None)

        ap_dist, an_dist = mat[p], mat[n]
        triplet_margin = ap_dist - an_dist if self.distance.is_inverted else an_dist - ap_dist

        self.set_stats(ap_dist, an_dist, triplet_margin)

        threshold_condition = triplet_margin <= self.margin
        if self.type_of_triplets == "hard":
            threshold_condition &= triplet_margin <= 0

        elif self.type_of_triplets == "semihard":
            threshold_condition &= triplet_margin > 0

        # filter out indices
        p_idx, n_idx = p[threshold_condition], n[threshold_condition]
        print(f"\np_idx: #{len(p_idx)}\n")

        return torch.arange(len(p_idx)), p_idx, n_idx

    def set_stats(self, ap_dist, an_dist, triplet_margin):

        if self.collect_stats:

            with torch.no_grad():

                self.pos_pair_dist = torch.mean(ap_dist).item()
                self.neg_pair_dist = torch.mean(an_dist).item()
                self.avg_triplet_margin = torch.mean(triplet_margin).item()


class FixedAnchorPairMiner(BaseMiner):

    def __init__(self, distance, anchor_emb: torch.Tensor, pos_margin: float = 0.2, neg_margin: float = 0.8,
                 batch_size: int = 32,  **kwargs):
        super(FixedAnchorPairMiner, self).__init__(**kwargs)

        self.p_margin = pos_margin
        self.n_margin = neg_margin
        self.distance = distance
        self.batch_size = batch_size
        self.anchor_emb = anchor_emb
        self.add_to_recordable_attributes(
            list_of_names=["pos_margin", "neg_margin"], is_stat=False
        )
        self.add_to_recordable_attributes(
            list_of_names=["pos_pair_dist", "neg_pair_dist"], is_stat=True
        )

    def mine(self, embeddings, labels, ref_emb=None, ref_label=None):

        # compute distances
        print(embeddings.shape)
        print(self.anchor_emb.shape)
        mat = self.distance(embeddings, self.anchor_emb)

        # get all pairs
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, None)

        pm, nm = a1 == a1, a2 == a2
        pi, ni = p[pm], n[nm]
        pos_pairs, neg_pairs = mat[pi], mat[ni]

        self.set_stats(pos_pairs, neg_pairs)

        # apply margin thresholds
        pos_mask = (pos_pairs < self.p_margin if self.distance.is_inverted else pos_pairs > self.p_margin)
        neg_mask = (neg_pairs > self.n_margin if self.distance.is_inverted else neg_pairs < self.n_margin)

        # compute final indices for each class
        p_idx, n_idx = p[pos_mask], n[neg_mask]

        print(f"\np_idx: #{len(p_idx)}\nn_idx: #{len(n_idx)}\n")

        return (torch.arange(len(p_idx)), p_idx,
                torch.arange(len(n_idx)), n_idx)

    def set_stats(self, pos_pair, neg_pair):

        if self.collect_stats:

            with torch.no_grad():

                self.pos_pair_dist = (
                    torch.mean(pos_pair).item() if len(pos_pair) > 0 else 0
                )

                self.neg_pair_dist = (
                    torch.mean(neg_pair).item() if len(neg_pair) > 0 else 0
                )
