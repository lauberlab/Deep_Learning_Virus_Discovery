import io
import sys
import zstandard as zstd
import numpy as np
import itertools
import argparse as ap
import traceback
import pandas as pd
import os
import gc
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

np.random.seed(1310)
PALMSCAN_DIR = "/mnt/twincore/ceph/cvir/user/nico/base_data/raw/palmscan2/"


def flags():
    parser = ap.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--out", type=str, required=False)
    parser.add_argument("--head", type=str, required=True)
    parser.add_argument("--cpus", type=int, default=12)
    parser.add_argument("--version", type=str, default="2")
    parser.add_argument("--redux", type=str, default=None, choices={"pca", "svd", "variance"})
    parser.add_argument("--redux_dim", type=int, default=156)
    parser.add_argument("--sse", action="store_true", default=False)
    parser.add_argument("--calpha", action="store_true", default=False)
    parser.add_argument("--only_pos", action="store_true", default=False)
    parser.add_argument("--only_neg", action="store_true", default=False)
    parser.add_argument("--num_batches", type=int, default=1, help="batching.")
    parser.add_argument("--min_num_nodes", type=int, default=9, help="number of kernels used.")
    parser.add_argument("--min_plddt", type=int, default=50, help="plddt-score threshold.")
    parser.add_argument("--masking", action="store_true", default=False, help="masking motifs.")
    parser.add_argument("--compressed", action="store_true", default=False, help="compressed npz-files.")

    return parser.parse_args()


def dimension_reduction(embedding, method, target_dim):

    scaler = StandardScaler()

    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)

    if not isinstance(embedding, np.ndarray) or embedding.dtype != np.float64:
        embedding = np.array(embedding, dtype=np.float64)

    if embedding.shape[1] < 1024:
        raise ValueError(f"given dim {embedding.shape[1]} < 1024!")

    # apply dimension reduction method
    if method == "pca":
        model = PCA(n_components=target_dim)
        transformed_sample = model.fit_transform(embedding)
        # explained_variance.append(model.explained_variance_ratio_.sum())

    elif method == "svd":
        model = TruncatedSVD(n_components=target_dim)
        transformed_sample = model.fit_transform(embedding)
        # explained_variance.append(model.explained_variance_ratio_.sum())

    elif method == "variance":
        var_thresh = VarianceThreshold(threshold=0.01)
        var_thresh.fit(embedding)
        top_features_idx = np.argsort(var_thresh.variances_)[-target_dim:]
        transformed_sample = embedding[:, top_features_idx]
        # explained_variance.append(None)  # Variance selection doesn't have variance ratios

    else:
        raise ValueError("Invalid method. Choose 'pca', 'svd', or 'variance'.")

    normalized_data = scaler.fit_transform(transformed_sample)

    return normalized_data


def calpha_coordinates(in_coordinates: np.ndarray, as_array: bool = False):
    # since Ca is always the second atom
    if as_array:
        return np.round(in_coordinates[1, :], decimals=3).astype(float)

    else:

        x_ca, y_ca, z_ca = np.round(in_coordinates[1, :], decimals=3)
        return (x_ca, y_ca, z_ca)


def mean_coordinates(in_coordinates: np.ndarray, as_array: bool = False, reduced: bool = False):
    # since x, y, z are considered columns in the given coordinates array
    # calculate the column-wise mean values
    in_coordinates = in_coordinates[:3, :] if reduced else in_coordinates

    x_mean, y_mean, z_mean = np.round(np.mean(in_coordinates, axis=0), decimals=3)
    out_coord = (x_mean, y_mean, z_mean)

    if as_array:
        out_coord = np.asarray(out_coord).astype(float)

    return out_coord


def define_mask(res_idx, motif_domain):

    res_mask_overlap = len(np.intersect1d(res_idx, motif_domain))
    res_mask_ratio = res_mask_overlap / len(res_idx)

    return 0 if res_mask_ratio >= 0.5 else 1


def sampling(input_data, sse_representation: bool = False):

    # define output container
    coordinates, feat, heads, labels, masks, skipped = [], [], [], [], [], []
    print(f"sampling...")

    try:

        if ARGS["masking"]:

            # get motif A-C information for masking
            # currently this points towards RdRp samples that have been cut to the motifs; there is no actual file
            # containing complete RdRp samples and their respective motif positions (to be implemented!)
            palmscan_file = f"{PALMSCAN_DIR}rdrp_motifs_v{ARGS['version']}.csv"
            palm_df = pd.read_csv(palmscan_file, sep=",", header=0)
            motif_names = palm_df["header"].tolist()
            motif_asta, motif_aend = palm_df["start_a"].tolist(), palm_df["end_a"].tolist()
            motif_bsta, motif_bend = palm_df["start_b"].tolist(), palm_df["end_b"].tolist()
            motif_csta, motif_cend = palm_df["start_c"].tolist(), palm_df["end_c"].tolist()

        # iterate over all elements per unpacked graph sample
        for idx, (header, atoms, resids, ress, atom_pos, ssids, sses, embs, label) in enumerate(input_data):

            # if idx == 1000:
            #    break

            header, label = str(header), int(label)
            print(f"\tit.{idx + 1} -- {header} (@class {label})")

            # extract masking information in case its flagged
            if ARGS["masking"] and label == 1:

                # set mask modifier --> randomly select Motif A, B or C
                mask_modifier = np.random.randint(0, 3)
                print(f"\tset mask modifier: {mask_modifier}")

                # check whether the current RdRp samples has confirmed motifs A-C
                if header in motif_names:

                    # get corresponding motif indices for masking
                    motif_idx = motif_names.index(header)

                    mask_motif_config = {0: list(range(motif_asta[motif_idx], motif_aend[motif_idx])),
                                         1: list(range(motif_bsta[motif_idx], motif_bend[motif_idx])),
                                         2: list(range(motif_csta[motif_idx], motif_cend[motif_idx]))}

                    motif_domain = mask_motif_config[mask_modifier]
                    print(f"\tset mask ranges for {header}.")

                else:
                    print(
                        f"\t'masking' was flagged for RdRp sample, but not valid motif was found for {header} --> skip")
                    continue

            else:
                motif_domain = [-1]
            
            # DIMENSIONALITY REDUCTION (PCA)
            # -------------------------------------------------------------------------------------------------------- #
            if ARGS["redux"] is not None:
                
                if embs.shape[0] >= ARGS["redux_dim"]:
                    embs = dimension_reduction(embs, ARGS["redux"], ARGS["redux_dim"])
                
                else:
                    print(f"embedding to small for dim reduction: {embs.shape[0]}!")
                    continue

                if embs.shape[-1] != ARGS["redux_dim"]:
                    print(f"embedding dim redux mismatch: {embs.shape[-1]} != {ARGS['redux_dim']}")
                    continue

            # MAPPING ATOMS TO RESIDUES
            # -------------------------------------------------------------------------------------------------------- #
            # get number of atoms per residue based on the unique residue number (ascending)
            atom_range = [np.sum(resids == rid) for rid in set(resids)]

            # reduce residues to every single occurrence as in the sequence
            res_redux = np.asarray(
                [f"{ress[i]}:{str(resid + 1)}" for i, resid in enumerate(resids) if i == 0 or resid != resids[i - 1]])
            resid_redux = np.asarray([resid for i, resid in enumerate(resids) if i == 0 or resid != resids[i - 1]])

            assert resid_redux.shape[0] == resid_redux.shape[0]

            # mapping start and end atoms for each residue & retrieve the corresponding position attributes
            res_atom_map = [(sum(atom_range[:i]), sum(atom_range[:i + 1])) for i in range(len(atom_range))]
            res_atom_pos = np.asarray([np.asarray(atom_pos[sta:end].astype(float)) for sta, end in res_atom_map],
                                      dtype=object)

            # compute centroids per residue based on the mean coordinates of the atoms it comprises
            if ARGS["calpha"]:
                res_pos = np.asarray([calpha_coordinates(pos) for i, pos in enumerate(res_atom_pos)])

            else:
                res_pos = np.asarray([mean_coordinates(pos, reduced=True) for i, pos in enumerate(res_atom_pos)])

            # check if attribute reduction was executed correctly; the number of centroids must be equal to the reduced
            # number of residues (sequence) and further match the amount of SSE elements given by DSSP
            if res_pos.shape[0] != res_redux.shape[0]:
                print(f"ERROR\tnumber of residues != number of centroids; mapping of atoms per residue failed!")
                skipped.append(header)
                print()
                continue

            elif res_pos.shape[0] != ssids.shape[0] != len(sses):
                print(f"ERROR\tnumber of centroids != number of SSE elements; incorrect mapping!")
                skipped.append(header)
                print()
                continue

            del res_atom_pos, res_atom_map, atom_range

            # use SSE representation (if flagged)
            if sse_representation:

                # MAPPING RESIDUES TO SSE
                # ---------------------------------------------------------------------------------------------------- #
                # reduce the SSE elements by counting their consecutive occurrence in 'sses'
                sse_redux = [(k + str(i + 1), int(len(list(g)))) for i, (k, g) in enumerate(itertools.groupby(sses))]

                # extract SSE attributes
                sse_ss, res_range = np.asarray(list(zip(*sse_redux))[0]), np.asarray(list(zip(*sse_redux))[1])

                if sum(res_range) != res_redux.shape[0]:
                    print(f"ERROR\tresidue number mismatch => {sum(res_range)} != {res_redux.shape[0]}!")
                    skipped.append(header)
                    print()
                    continue

                # map residues to SSE similar to atoms per residue
                sse_res_map = [(sum(res_range[:i]), sum(res_range[:i + 1])) for i in range(len(res_range))]
                sse_res_pos = [res_pos[sta:end] for sta, end in sse_res_map]

                # compute the SSE centroids
                sse_pos = np.asarray([mean_coordinates(pos) for i, pos in enumerate(sse_res_pos)])

                del sse_res_map, sse_res_pos, sse_redux

                if sse_pos.shape[0] != sse_ss.shape[0]:
                    print(f"ERROR\tmismatching number of SSE!")
                    skipped.append(header)
                    print()
                    continue

                # PROCESSING SSE
                # ---------------------------------------------------------------------------------------------------- #
                sse_pos, sse_nodes, sse_res, sse_embs, mask, aa_idx, discard = [], [], [], [], [], 0, 0

                # convert into SSE form
                print(f"\treducing graph to SSEs ..")
                for sid, sse in enumerate(sse_ss):

                    # get number of residues of current SSE element
                    num_residues = res_range[sid]

                    # print(f"sse: {sse} -- residues: {num_residues}")

                    # skip coils since they are regions absent of any SSE
                    if sse[0] == "C" or num_residues == 1:
                        discard += 1
                        continue

                    # define 'N' base on the residue number of current SSE
                    if num_residues > 5:
                        N = 4

                    else:
                        N = num_residues - 1

                    # reduce the residue positions array according
                    aa_ide = aa_idx + num_residues
                    sse_pos_range = res_pos[aa_idx:aa_ide]

                    # get residue IDs and corresponding residues per SSE
                    current_res_rex = res_redux[aa_idx:aa_ide]
                    current_resid_rex = resid_redux[aa_idx:aa_ide]

                    # Mask specific SSE that are part of either motif A, B or C
                    mask_binary = define_mask(res_idx=current_resid_rex, motif_domain=motif_domain)

                    # append
                    mask.append(mask_binary)
                    sse_res.append(current_res_rex)
                    sse_nodes.append(sse)

                    # get embeddings and sum up the embedding to a single vector of size (1, 1024)
                    """
                    summation of the SSE embeddings might result in information loss! instead we proceed with aa embeddings
                    per SSE as node features (M, 1024) where M correlates to the number of aa belonging to the respective SSE
                    """
                    sse_emb = embs[aa_idx:aa_ide]
                

                    """
                    if ARGS["redux"] is not None and sse_emb.shape[0] >= ARGS["redux_dim"]:
                        sse_emb = dimension_reduction(sse_emb, ARGS["redux"], ARGS["redux_dim"])

                        if sse_emb.shape[-1] != ARGS["redux_dim"]:
                            print(f"embedding dim redux mismatch: {sse_emb.shape[-1]} != {ARGS['redux_dim']}")
                            continue
                    """
                    sse_embs.append(sse_emb)

                    # compute p and p & append to variable holding the positional space vectors
                    if N > 1:
                        sse_pos.append(mean_coordinates(sse_pos_range, as_array=True))

                    else:
                        sse_pos.append(sse_pos_range[0])

                    # increase counters for aa residues (index)
                    aa_idx += num_residues

                del res_range

                # convert to numpy
                features = np.array(sse_embs, dtype=object)
                mask = np.array(mask, dtype=object)
                positions = np.array(sse_pos, dtype=object)

                # discard filter
                if features.shape[0] < ARGS["min_num_nodes"]:
                    print(f"ERROR\tinsufficient number of nodes\n\t{features.shape[0]}")
                    skipped.append(header)
                    continue

                if features.shape[0] != mask.shape[0]:
                    print(f"ERROR\tdimensions mismatch on raw node features\n\t{features.shape[0]} != {mask.shape[0]}")
                    skipped.append(header)
                    print()
                    continue

                print(f"\tset node features.")

            # aa representation (Default)
            else:
                """
                if ARGS["redux"] is not None and embs.shape[0] >= ARGS["redux_dim"]:
                    embs = dimension_reduction(embs, ARGS["redux"], ARGS["redux_dim"])

                    if embs.shape[-1] != ARGS["redux_dim"]:
                        print(f"embedding dim redux mismatch: {embs.shape[-1]} != {ARGS['redux_dim']}")
                        continue
                """
                features = np.array(embs, dtype=object)
                mask_binary = define_mask(res_idx=resid_redux, motif_domain=motif_domain)
                mask = np.array(mask_binary, dtype=object)
                positions = np.array(res_pos, dtype=object)

            del resid_redux, res_pos, res_redux

            # ATTACHMENT
            # -------------------------------------------------------------------------------------------------------- #
            coordinates.append(positions)
            feat.append(features)
            heads.append(header)
            masks.append(mask)
            labels.append(label)
            print()

            del positions, mask, label, header, features
            gc.collect()

        return coordinates, feat, heads, masks, labels, skipped

    except Exception as e:
        traceback.print_exc()
        sys.exit(-1)


if __name__ == "__main__":

    print()

    # get user input
    ARGS = vars(flags())

    # where to store data
    base_dir = "/mnt/twincore/ceph/cvir/user/nico/sse_graphs/"
    data_dir = base_dir if ARGS["out"] is None else ARGS["out"]

    if ARGS["head"] is not None:

        # define input & output file
        input_file = ARGS["inp"]
        output_file = data_dir + ARGS["head"]

        # don't overwrite existing data (remove manually beforehand, in case you want to rerun this script)
        if not os.path.exists(output_file):

            # load data & chunking
            if not ARGS["compressed"]:
                data = np.load(input_file, allow_pickle=True)

            else:
                with open(input_file, "rb") as compressed_file:
                    dctx = zstd.ZstdDecompressor()

                    with dctx.stream_reader(compressed_file) as reader:
                        decompressed_bytes = io.BytesIO(reader.read())

                        data = np.load(decompressed_bytes, allow_pickle=True)

            num_samples = len(data["h"].tolist())
            print(f"loaded #{num_samples} samples..")

            # filter step
            graph_plddt = data["t"] if "t" in list(data.keys()) else np.asarray(
                [100.00] * num_samples)

            # based on plddt-scores
            plddt_idx = np.where(graph_plddt >= ARGS["min_plddt"])[0]

            # unpack input data
            graph_header = data["h"][plddt_idx]
            graph_atoms = data["a"][plddt_idx]
            graph_positions = data["p"][plddt_idx]
            graph_residue_ids = data["i"][plddt_idx]
            graph_residues = data["r"][plddt_idx]
            graph_sse_ids = data["s"][plddt_idx]
            graph_sse = data["e"][plddt_idx]
            graph_embeddings = data["q"][plddt_idx]
            graph_labels = data["l"][plddt_idx]

            print(f"extracted data attributes..")
            del data

            pos_idx = np.where(graph_labels == 1)[0]
            neg_idx = np.where(graph_labels == 0)[0]
            default = np.arange(0, len(graph_labels))

            if ARGS["only_pos"]:
                indices = pos_idx

            elif ARGS["only_neg"]:
                indices = neg_idx

            else:
                indices = default

            # batching
            num_batches = ARGS["num_batches"]
            batch_size = num_samples // num_batches

            if num_samples % num_batches != 0:
                num_batches += 1

            batch_sta, batch_end = 0, batch_size

            for i in range(num_batches):

                # zip unpacked input data
                iterate_data = zip(
                    [head.replace(".", "_") for head in graph_header[batch_sta:batch_end][indices]],
                    graph_atoms[batch_sta:batch_end][indices],
                    graph_residue_ids[batch_sta:batch_end][indices],
                    graph_residues[batch_sta:batch_end][indices],
                    graph_positions[batch_sta:batch_end][indices],
                    graph_sse_ids[batch_sta:batch_end][indices],
                    graph_sse[batch_sta:batch_end][indices],
                    graph_embeddings[batch_sta:batch_end][indices],
                    graph_labels[batch_sta:batch_end][indices],
                )

                batch_sta = batch_end
                batch_end = batch_sta + batch_size

                print(f"processing batch {i+1}:")

                # retrieve graph samples
                pos_data, feature_data, header_data, mask_data, label_data, skip_data = sampling(iterate_data, ARGS["sse"])

                del iterate_data
                print(f"finished sampling batch {i+1}..")
                print(f"skipped {len(skip_data)}")
                print()

                # convert to numpy
                np_pos = np.array(pos_data, dtype=object)
                np_feat = np.array(feature_data, dtype=object)
                np_head = np.array(header_data, dtype=object)
                np_mask = np.array(mask_data, dtype=object)
                np_label = np.array(label_data, dtype=object)

                assert np_pos.shape[0] == np_feat.shape[0] == np_head.shape[0] == np_label.shape[0] == np_mask.shape[0]
                del pos_data, feature_data, header_data, mask_data, label_data, skip_data

                data_size = np_pos.shape[0]

                if data_size == 0:

                    print(f"dataframe is empty ==> saving not possible")
                    print()

                else:

                    batched_output = output_file + f"_0{i+1}.npz"

                    # save remaining samples
                    np.savez(batched_output, p=np_pos, x=np_feat, h=np_head, m=np_mask, y=np_label)
                    print()

                    del np_pos, np_feat, np_head, np_mask, np_label

                gc.collect()

        else:
            print(f"{output_file} already exists..")

    else:
        sys.exit(-1)
