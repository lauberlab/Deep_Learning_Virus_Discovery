import pickle
import logging
import random

import pandas as pd
import numpy as np
from pathlib import Path
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from model.autoencoder import SSEGraphEncoderModel
from dataset.graph_set import GraphSubset
from dataset.graph_set import BatchData
import sys


TEST_SAMPLE_ORIGIN_TABLE = {
    1: "Protonido", 2: "Nidovirales", 3: "DNA-dep.DNA-poly", 4: "RNA-dep.DNA-poly",
}

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


# FUNCTION [protected]: move batch tensors to target device
def move_batch_to_device(batch: BatchData, device: torch.device):
    if isinstance(batch.x, torch.Tensor):
        batch.x = batch.x.to(device)

    if isinstance(batch.p, torch.Tensor):
        batch.p = batch.p.to(device)

    if isinstance(batch.m, torch.Tensor):
        batch.m = batch.m.to(device)

    if isinstance(batch.e, torch.Tensor):
        batch.e = batch.e.to(device)

    if isinstance(batch.y, torch.Tensor):
        batch.y = batch.y.to(device)

    return batch


def reshape_labels(criterion, yh, yt):
    if isinstance(criterion, nn.BCELoss) or isinstance(criterion, nn.BCEWithLogitsLoss):
        yh = yh.view(-1)

    return yh, yt.view(-1)


def get_checkpoint(from_checkpoint: bool, model_dir: str, rank: int):

    if from_checkpoint:
        checkpoint = torch.load(f"{model_dir}/checkpoints/0{rank}_model_check.pth")

    else:
        checkpoint = torch.load(f"{model_dir}/checkpoints/0{rank}_model_final.pth")

    return checkpoint


def load_model_params(model_dir: str, model_rank: int):

    # load_model_ranking = load_from_pickle(model_dir + "/ranking.pkl")
    # max_rank = load_model_ranking[0][0] if model_rank is None else model_rank
    logging.info(f"load model params k={model_rank}")

    # load params
    load_params = load_from_pickle(model_dir + "/params.pkl")
    load_params["model_rank"] = model_rank

    return load_params, get_checkpoint(load_params["load_from_checkpoint"], model_dir, model_rank)


def transform_similarity(results: pd.DataFrame):

    smallest_positive_value = sys.float_info.epsilon

    # unpack
    similarities, labels = results["y"], results["labels"]
    c0_idx, c1_idx = np.where(labels == 0)[0], np.where(labels == 1)[0]
    c0_sim, c1_sim = similarities[c0_idx], similarities[c1_idx]

    # get TP / FN
    tp = len([i for i, cs in enumerate(c1_sim) if cs > 0.0 + smallest_positive_value])
    fn = len([i for i, cs in enumerate(c1_sim) if cs < 0.0 + smallest_positive_value])

    # get TN / FP
    tn = len([i for i, cs in enumerate(c0_sim) if cs < 0.0 - smallest_positive_value])
    fp = len([i for i, cs in enumerate(c0_sim) if cs > 0.0 - smallest_positive_value])

    # precision
    # what proportion of positive identifications was actually correct?
    # model with no FP has a precision of 1.0
    precision = tp / (tp + fp)

    # recall --> True Positive Rate (TPR)
    # what proportion of actual positives was identified correctly?
    # model with no FN has recall of 1.0
    tpr = tp / (tp + fn)

    # False Positive Rate (FPR)
    fpr = np.array(fp / (fp + tn))

    # AUC for single threshold @ 0.0, considering the points (0,0), (FPR, TPR), and (1,1)
    auc = 0.5 * (tpr * (1 - fpr) + fpr * (1 - tpr) + tpr)

    # compute F1 score
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

    return precision, tpr, fpr, auc, f1


def shuffle(indices: list, seed_value: int):

    np.random.seed(seed_value)
    return np.random.permutation(indices)


def split_pos_neg(dataset: list):

    # split for positive and negative samples
    positive_samples = [sample for sample in dataset if sample.y == 1]
    negative_samples = [sample for sample in dataset if sample.y == 0]

    return positive_samples, negative_samples


def merge_npz_files(file_paths):
    merged_data = {}

    for file_path in file_paths:
        with np.load(file_path, allow_pickle=True) as data:

            for key in data.files:
                if key in merged_data:
                    merged_data[key] = np.concatenate((merged_data[key], data[key]), axis=0)

                else:
                    merged_data[key] = data[key]

    return merged_data


def model_init(**model_params):

    # unpack general parameters
    use_seed = model_params["seed_value"]

    model = SSEGraphEncoderModel(seed=use_seed,
                                 device=model_params["device"],
                                 x_channels=model_params["x_dim"],
                                 edge_channels=model_params["edge_channels"],
                                 hidden_channels=model_params["hidden_channels"],
                                 num_attn_heads=model_params["attention_heads"],
                                 pooling=model_params["pooling"],
                                 num_layers_enc=model_params["num_enc_layers"],
                                 motif_masking=model_params["motif_masking"],
                                 final_mlp=model_params["final_mlp"],
                                 augment_eps=model_params["augment_eps"],
                                 num_min_neighbours=model_params["num_min_neighbours"],
                                 neighbour_fraction=model_params["neighbour_fraction"],
                                 num_pos_features=model_params["num_pos_features"],
    )

    # check for pretrain-flag
    if model_params["pretrain"]:

        # load weights from autoencoder
        at_checkpoint = "./artifacts/autoencoder/Autoencoder_01/at_check.pth"
        at_weights = torch.load(at_checkpoint)
        model.load_state_dict(at_weights["model_state_dict"])

        for param in model.parameters():
            param.requires_grad = False

    # cast to predefined device
    print(f"new model initiated..")
    # print(f"{model}\n")  # print model summary

    return model


def permute_labels(dataset):

    """Randomly shuffle labels in a dataset."""
    permuted_y = dataset.y.clone()
    idx = torch.randperm(len(permuted_y))
    permuted_y = permuted_y[idx]

    return permuted_y


def k_fold_train_test_split(dataset, p: int, k: int = 5, test_size: float = 0.1, seed_value: int = 42,
                            label_permutation: bool = False):

    # Lists to store results across folds
    train_folds, test_folds = [], []

    # test splits based on indices
    pos_indices = np.where(dataset.y == 1)[0]
    neg_indices = np.where(dataset.y == 0)[0]

    # P-randomizer
    p_rng = random.Random(p)
    p_rnd = p_rng.randint(0, 100) + 13

    for idx in range(k):

        # set up a randomizer
        rng = np.random.default_rng(seed_value + idx + p_rnd)

        # positive class: data split
        pos_split = int(len(pos_indices) * (1 - test_size))
        pos_tr_idx, pos_te_idx = pos_indices[:pos_split], pos_indices[pos_split:]

        # negative class: data split
        neg_split = int(len(neg_indices) * (1 - test_size))
        neg_tr_idx, neg_te_idx = neg_indices[:neg_split], neg_indices[neg_split:]

        # assemble train set
        train_indices = np.concatenate((pos_tr_idx, neg_tr_idx))
        rng.shuffle(train_indices)
        train_set = GraphSubset(dataset, train_indices)

        if label_permutation:
            train_set.y = permute_labels(train_set)

        # assemble test set
        test_indices = np.concatenate((pos_te_idx, neg_te_idx))
        rng.shuffle(test_indices)
        test_set = GraphSubset(dataset, test_indices)

        # prepare training fold
        train_folds.append(train_set)

        # prepare test fold
        test_folds.append(test_set)

    return train_folds, test_folds


# ---------------------------------------------------------------------------------------------------------------------#
# get max seq len among all data sets
def create_target_dir(path: str):

    import os.path
    import subprocess as sp

    if not os.path.exists(path):

        cmd = ["mkdir", path]
        process = sp.Popen(cmd)
        process.wait()


# ---------------------------------------------------------------------------------------------------------------------#
# get max seq len among all data sets
def get_max_seq_len(datasets: list):

    merge = list()

    for ds in datasets:

        ds = [x.shape[1] for x in ds]

        merge.extend(ds)

    return max(merge)


# ---------------------------------------------------------------------------------------------------------------------#
# pickle saving and loading
def save_with_pickle(data, path):

    with open(path, "wb") as tar:
        pickle.dump(data, tar)


def load_from_pickle(path):

    with open(path, "rb") as src:

        return pickle.load(src)


# ---------------------------------------------------------------------------------------------------------------------#
def get_template(template_location: str, _from: str):

    template_files = [str(p) for p in Path(template_location).glob("*_template.npz")]
    template_headers = [tf.split("/")[-1].split("_")[1] for tf in template_files]

    if len(template_files) >= 1:

        # use specified template
        if _from is not None and _from in template_headers:
            _idx = template_headers.index(_from)
            return np.load(template_files[_idx], allow_pickle=True)

        else:
            print(f"[TEMPLATE] '{_from}' not found! shut down!")
            sys.exit(-1)

    else:
        raise ValueError(f"'template_files' is empty: '{template_files}'. Please, provide template file(s).")


# ---------------------------------------------------------------------------------------------------------------------#
def meta_from_loader(data_loader):

    labels, headers = [], []

    for batch in data_loader:
        labels.extend(batch.y.numpy())

        header_indices = batch.head_len.tolist()
        headers_long = batch.head.tolist()

        sta_idx, end_idx = 0, 0
        for header_idx in header_indices:
            end_idx += header_idx
            header_integer = headers_long[sta_idx:end_idx]
            headers.append("".join([chr(c) for c in header_integer]))

            sta_idx += header_idx

    return np.asarray(labels), np.asarray(headers)


# ---------------------------------------------------------------------------------------------------------------------#
def train_test_splitter(X, y, size, seed):

    # split into training, validation and test sets using scikit-learn
    # train-test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, stratify=y, random_state=seed)

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------------------------------------------------#
# function to select and return a final activation function
class ActivationCriterionSelection:

    def __init__(self, choice):
        super(ActivationCriterionSelection, self).__init__()

        act, crit = self.__selection_func(choice)

        self.__activation = act
        self.__criterion = crit

    @staticmethod
    def __selection_func(choice: str):

        # for binary classification
        if choice == "sigmoid":

            return None, nn.BCELoss()

        # for multi-class classification
        elif choice == "softmax":

            return None, nn.CrossEntropyLoss()

        # for multi-label classification
        else:

            return None, nn.BCEWithLogitsLoss()

    def get_activation(self):

        return self.__activation

    def get_criterion(self):

        return self.__criterion


def NIDO_RdRp_domain(target_header, hmm_data: pd.DataFrame, metacontigs: bool = False):

    # extract HMM hits from provided data sheet
    targets, tags = hmm_data["target_name"].tolist(), hmm_data["tag"].tolist()

    starts, ends = np.asarray(hmm_data["start"].tolist()).astype(int), np.asarray(hmm_data["end"].tolist()).astype(int)
    ranges = np.asarray(hmm_data["range"].tolist()).astype(int)

    try:
        # find HMM search hit range for exploration sample
        target_idx = targets.index(target_header)
        target_start, target_end = starts[target_idx], ends[target_idx]

        if metacontigs:
            target_start, target_end = 1, ranges[target_idx]

        return [target_start, target_end]

    except ValueError as e:

        logging.warning(f"\t{target_header} not found in provided HMM hit list.")
        return [0, 0]


def rescaling(values: list, min_val: float, max_val: float):

    return np.asarray([-1 if value == 0 else round((value - min_val) / (max_val - min_val), 2) for value in values],
                      dtype=float)


def median_sample(samples):

    sorted_samples = sorted(samples)
    n = len(sorted_samples)

    if n % 2 == 1:

        # in case the group has an odd number of members
        return sorted_samples[n // 2]

    else:

        # in case the group has an even number of members, return the average of both middle elements
        n1, n2 = n // 2 - 1, n // 2

        return sorted_samples[(n1 + n2) // 2]


def sample_score(data, d_type: str, t: float, group_range: int, weight_mad: float):

    top_group_score, top_group_idx, top_group = 0, None, []

    # collect all samples that are exceeding a given distance/similarity threshold and apply sorting for the same
    if d_type == "distance":
        top_samples = sorted([[mad, distance, sample] for mad, distance, sample in data if distance <= t],
                             key=lambda x: x[-1])

    else:
        top_samples = sorted([[mad, similarity, sample] for mad, similarity, sample in data if similarity >= t],
                             key=lambda x: x[-1])

    # convert to numpy
    top_samples = np.asarray(top_samples)

    # specify return values
    default_return = [top_group_idx, top_group_score, top_group]

    # determining MAD influence on overall scoring
    weight_distance = 1 - weight_mad

    if top_samples.size != 0:

        top_mad, top_distances, top_indices = top_samples[:, 0], top_samples[:, 1], top_samples[:, 2]

        # form distance-groups based on the corresponding sample indices
        # groups must have more than one member, so there can be empty groups!
        groups = find_groups(mads=top_mad, indices=top_indices, distances=top_distances, group_range=group_range)

        # check if groups exist, because of how groups are defined
        if len(groups) > 0:

            logging.info(f"\t#{len(groups)} --> computing scores")

            # compute sample scores for all group members for each group
            for group in groups:

                # unpack arrays
                group_mad, group_dist, group_sample = group

                # assert group_mad.shape[0] == group_dist.shape[0] == group_sample.shape[0]
                assert len(group_mad) == len(group_dist) == len(group_sample)

                # determine group length
                group_len = len(group_mad)

                # define reciprocal coefficient 'C', where k equals range in where to find neighbouring samples (def: 100)
                # used to weight the average score per group; decreases the influence of smaller groups in relation to
                # bigger ones
                C = 1 - (group_range / (group_len + group_range))

                group_score, samples = [], []
                for mad, dist, sample in zip(group_mad, group_dist, group_sample):

                    # score = ((wd * dist) - (wm * (1 - mad))) * C
                    local_score = ((weight_distance * dist) - (weight_mad * (1 - mad)))
                    group_score.append(local_score)
                    samples.append(sample)

                group_score = ((sum(group_score) / len(group_score)) * C) * 2
                sample_index = median_sample(samples)

                if group_score > top_group_score:
                    top_group_score = group_score
                    top_group_idx = sample_index
                    top_group = samples

            logging.info(f"\tindex | score:  {top_group_idx} | {top_group_score}")

            default_return = [top_group_idx, top_group_score, top_group]

    return default_return


def trapezoidal_rule(x: np.ndarray, y: np.ndarray):

    r"""
    find/approximate the value of a definite integral or simply suggesting the area under the curve by dividing it
    into several trapezoids controlled by param 'n'. Their sum is found to get the area of the curve. Assuming, we have
    a continuous curve defined on closed interval [x0, xN]; we can now divide this closed interval into n equal
    sub-intervals with each having the width of delta_x = (x0 -xN) / n such that x0 = x0 < x1 < x2 < ... < xN

    the area under cuve can now be computed as (delta_x / 2) [y0 + 2(y1 + y2 + ... yN-1) + yN], where y0, .., yN are the
    values at x1, x2, .., xN respectively

    since we are using coordinates instead of a function, y is already defined and does not need to be computed as values
    of function x (as originally implemented in the trapezoidal rule)
    """

    n = max(x.shape[0] - 1, 1)

    # grid spacing ==> h=delta_x
    h = (x[-1] - x[0]) / n

    # calculate area of first and last trapezoids ==> (delta_x / 2) [y0 + yN]
    area = (h / 2.0) * (y[-1] + y[0])

    # calculating area of the middle trapezoids ==> 2(y1 + y2 + ... yN) * h=(dela_x/2)
    for i in range(1, n):
        area += 2 * (h * y[i])

    return area


# POSTPROCESSING: --> GROUPING
# -------------------------------------------------------------------------------------------------------------------- #
def easy_grouping(data: np.ndarray, t: float = 0.0):

    def __split(__samples, __vals):
        split_idx = [i for i in range(1, len(__samples)) if __samples[i] != __samples[i-1] + 1]

        return np.split(__samples, split_idx), np.split(__vals, split_idx)

    # filter for low 'x' (low similarity) and high 'x'
    init_samples = np.asarray(list(range(0, data.shape[0])))
    hx_samples, hd = init_samples[data > t], data[data > t]
    lx_samples, ld = init_samples[data <= t], data[data <= t]

    # form groups by splitting at non-consecutive order (per index)
    hx_sample_split, hd_split = __split(hx_samples, hd)
    lx_sample_split, ld_split = __split(lx_samples, ld)

    # merge to get a complete list of individual groups
    sample_split = hx_sample_split + lx_sample_split
    data_split = hd_split + ld_split

    assert len(sample_split) == len(data_split)

    # group_lens, avg_d, dd, samples = [], [], [], []
    out = {"s": [], "gl": [], "md": [], "d": [], "ph": []}
    for idx, ss in enumerate(sample_split):

        # similarity / distance
        d = data_split[idx]

        if len(ss) > 0:
            out["gl"].append(len(ss))
            out["s"].append(sample_split[idx])
            out["d"].append(d)

            # coefficient 'C' respects the group length
            C = 1 - (1 / (len(ss) + 0.05))
            md = round(float(sum(d) / len(ss)), 3) * C
            out["md"].append(md)
            out["ph"].append(1 if md > 0.0 else 0)

    return pd.DataFrame(data=out)


def group_curve_segments(x: np.ndarray, y: np.ndarray):

    assert y.shape[0] == x.shape[0]

    indices = np.where(y > 0)[0]
    split_idx = [i for i in range(1, len(indices)) if indices[i] != indices[i-1] + 1]

    y_vals = np.split(y[indices], split_idx)
    x_vals = np.split(x[indices], split_idx)

    return x_vals, y_vals


def sample_groups(mads, distances, sample_indices, group_range: int = 25, member: int = 3):

    groups, current_group = [], []

    for mad, distance, index in zip(mads, distances, sample_indices):

        # check if it's either the first group or the currently investigated hit can be assigned to the last group
        if not current_group or abs(index - current_group[0][-1]) <= group_range:

            # add to group
            current_group.append((mad, distance, index))

        else:

            # ensure groups have more than one member to be in the definition of a group
            if len(current_group) >= member:
                groups.append(current_group)

            # form a new group
            current_group = [(mad, distance, index)]

    # add the last group
    if current_group and len(current_group) >= member:
        groups.append(current_group)

    return groups


def find_groups(mads, distances, indices, group_range: int):

    groups = []

    if indices.shape[0] > 1:

        # get anchor indices
        fi, mi, la = indices[0], indices[len(indices) // 2], indices[-1]

        # reduces indices to first anchor (f)
        f_indices = np.where(abs(fi - indices) <= group_range)[0]
        # f_indices = indices[f_idx]
        groups.append((mads[f_indices], distances[f_indices], indices[f_indices]))

        m_indices, share_la_mi = None, 0.0

        # center anchor (m)
        if mi != la and mi != fi:
            m_indices = np.where(abs(mi - indices) <= (group_range // 2))[0]

            overlap_mi_fi = [idm for idm in m_indices if idm in f_indices]
            share_mi_fi = len(overlap_mi_fi) / len(m_indices)

            # in case overlap is low, we consider 'm' as its own group
            if share_mi_fi < 0.75:
                groups.append((mads[m_indices], distances[m_indices], indices[m_indices]))

            else:
                m_indices = None

        # final anchor (l)
        if la != fi and la != mi:
            l_indices = np.where(abs(la - indices) <= group_range)[0]

            # find overlap between last and first group
            overlap_la_fi = [idl for idl in l_indices if idl in f_indices]
            share_la_fi = len(overlap_la_fi) / len(l_indices)

            # find overlap between last and center group
            if m_indices is not None:
                overlap_la_mi = [idl for idl in l_indices if idl in m_indices]
                share_la_mi = len(overlap_la_mi) / len(l_indices)

            # in case overlap between the first and center group is low, 'l' is considered a group
            if share_la_fi < 0.75 and share_la_mi < 0.75:
                groups.append((mads[l_indices], distances[l_indices], indices[l_indices]))

    return groups
# -------------------------------------------------------------------------------------------------------------------- #

