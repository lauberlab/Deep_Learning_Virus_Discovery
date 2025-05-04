import sys

import numpy as np
from pytorch_metric_learning import distances, losses, miners, reducers
from .fixed_anchor_miner import FixedAnchorPairMiner, FixedAnchorTripletMiner
import torch


"""
# data --> sampler --> miner --> loss --> reducer --> final loss value


# reducers specify how to go from many losses to a single loss

# define triplet miner: this miner will find the hardest positive and negative samples within a batch, and use
# those to form a single triplet, so far batch size 'N' results in 'N' output triplets
"""


# loss func, miner
def get_pml_profile(profile: str, loss_margin: float, embedding_size: int, norm_embeddings: bool = True,
                    use_miner: bool = False):

    if profile == "TripletMargin":
        reducer = reducers.AvgNonZeroReducer()
        distance = distances.LpDistance(normalize_embeddings=norm_embeddings, power=1)

        criterion = losses.TripletMarginLoss(margin=loss_margin, distance=distance, reducer=reducer)
        miner = miners.TripletMarginMiner(margin=loss_margin, distance=distance, type_of_triplets="hard") \
            if use_miner else None

        dist_type = "distance"

        return criterion, miner, dist_type, None

    elif profile == "TripletBatchHard":
        reducer = reducers.AvgNonZeroReducer()
        distance = distances.LpDistance(normalize_embeddings=norm_embeddings, power=1)

        criterion = losses.TripletMarginLoss(margin=loss_margin, distance=distance, reducer=reducer)
        miner = miners.BatchHardMiner(distance=distance) if use_miner else None

        dist_type = "distance"

        return criterion, miner, dist_type, None

    elif profile == "ContrastivePairMargin":
        reducer = reducers.AvgNonZeroReducer()
        distance = distances.CosineSimilarity()
        criterion = losses.ContrastiveLoss(pos_margin=0.8, neg_margin=0.2, reducer=reducer, distance=distance)
        miner = miners.PairMarginMiner(pos_margin=0.8, neg_margin=0.2, distance=distance) if use_miner else None

        dist_type = "distance"

        return criterion, miner, dist_type, None

    elif profile == "ProxyAnchorTriplet":
        reducer = reducers.DivisorReducer()
        distance = distances.CosineSimilarity()
        criterion = losses.ProxyAnchorLoss(num_classes=2, embedding_size=embedding_size, reducer=reducer,
                                           margin=loss_margin, alpha=32, distance=distance)
        criterion_optim = torch.optim.SGD(criterion.parameters(), lr=0.01)
        miner = miners.TripletMarginMiner(margin=loss_margin, distance=distance, type_of_triplets="hard") \
            if use_miner else None

        dist_type = "similarity"

        return criterion, miner, dist_type, criterion_optim

    elif profile == "SoftTriple":
        reducer = reducers.MeanReducer()
        distance = distances.CosineSimilarity()
        criterion = losses.SoftTripleLoss(num_classes=2, embedding_size=embedding_size, reducer=reducer,
                                          margin=0.01, centers_per_class=10, gamma=0.15, distance=distance)
        criterion_optim = torch.optim.SGD(criterion.parameters(), lr=0.01)
        dist_type = "similarity"

        return criterion, None, dist_type, criterion_optim

    elif profile == "NTXentPair":
        reducer = reducers.MeanReducer()
        distance = distances.CosineSimilarity()
        criterion = losses.NTXentLoss(temperature=0.06, reducer=reducer, distance=distance)
        miner = miners.PairMarginMiner(pos_margin=0.8, neg_margin=0.2, distance=distance) if use_miner else None

        dist_type = "similarity"

        return criterion, miner, dist_type, None

    elif profile == "IntraPairMargin":

        reducer = reducers.MeanReducer()
        distance1 = distances.CosineSimilarity()
        distance2 = distances.LpDistance(normalize_embeddings=norm_embeddings, power=1)
        loss1 = losses.TupletMarginLoss(margin=5.73, scale=64, reducer=reducer, distance=distance1)
        loss2 = losses.IntraPairVarianceLoss(pos_eps=0.01, neg_eps=0.01, reducer=reducer, distance=distance2)
        criterion = losses.MultipleLoss([loss1, loss2], weights=[1, 0.5])
        miner = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8, distance=distance1)

        dist_type = "similarity"

        return criterion, miner, dist_type, None

    elif profile == "TupletPairMargin":

        reducer = reducers.MeanReducer()
        distance = distances.CosineSimilarity()
        criterion = losses.TupletMarginLoss(margin=5.73, scale=64, reducer=reducer, distance=distance)
        miner = miners.PairMarginMiner(pos_margin=0.2, neg_margin=0.8, distance=distance) if use_miner else None

        dist_type = "similarity"

        return criterion, miner, dist_type, None

    else:
        print(f"[WARNING] PML set for {profile} is not supported.")

        return None, None, None, None


def get_miner_with_fixed_anchor(profile: str, batch_size: int, loss_margin: float, template: np.ndarray):

    distance = distances.CosineSimilarity()

    if "Pair" in profile:
        miner = FixedAnchorPairMiner(pos_margin=0.2, neg_margin=0.8, distance=distance, batch_size=batch_size,
                                     anchor_emb=template)

    elif "Triplet" in profile:
        miner = FixedAnchorTripletMiner(margin=loss_margin, distance=distance, type_of_triplets="hard",
                                        batch_size=batch_size, anchor_emb=template)
    else:
        print(f"cannot identify 'FixedAnchorMiner'! check 'PML-profile' again..")
        sys.exit(-1)

    return miner
