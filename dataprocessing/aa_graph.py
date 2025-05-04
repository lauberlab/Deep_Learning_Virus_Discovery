import sys
import time

import numpy as np
import networkx as nx
import itertools
import argparse as ap
from itertools import combinations
import traceback
import pandas as pd
from pathlib import Path
from sse_utils import get_sparse_adjacency_matrix, get_distance_matrix, check_vector_direction


np.random.seed(1310)
EDGE_KIND_MAP = {"parallel": 0, "antiparallel": 1, "mixed": 2}


def flags():

    parser = ap.ArgumentParser()
    parser.add_argument("--inp", type=str, required=False)
    parser.add_argument("--save_plots", action="store_true", default=False)
    parser.add_argument("--test_head", type=str, default=None)
    parser.add_argument("--calpha", action="store_true", default=False)
    parser.add_argument("--distance", type=int, default=4, help="distance threshold between residues.")
    parser.add_argument("--min_num_nodes", type=int, default=9, help="number of kernels used.")
    parser.add_argument("--masking", action="store_true", default=False, help="masking motifs.")

    return parser.parse_args()


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


def compute_residue_contacts(positions: np.ndarray, atom_to_aa_map, aa_atom_keys: np.ndarray, max_dist: float = 4.0):

    # compute atom distances from their coordinates; resulting in a distance matrix
    atom_distance_mat = get_distance_matrix(coordinates=positions.astype(float))

    # output dict
    aa_contacts = {}
    stat_contacts = []

    # nested looping through all combinations of amino acid residues
    ctr_i = 0
    for idx, ati_pos_per_res in enumerate(atom_to_aa_map[:-1]):

        aai = aa_atom_keys[:-1][idx]                                # get the amino acid residue tag for idx
        ati = (ctr_i, ctr_i + ati_pos_per_res.shape[0])             # get the corresponding atoms for that residue

        ctr_j = 0
        for jdx, atj_pos_per_res in enumerate(atom_to_aa_map[idx+1:]):

            aaj = aa_atom_keys[idx+1:][jdx]                         # get the amino acid residue tag for jdx
            atj = (ctr_j, ctr_j + atj_pos_per_res.shape[0])

            # slicing the distance matrix based on the atom coordinates in 'pos_mapping'
            distance = atom_distance_mat[ati[0]:ati[1], atj[0]:atj[1]]
            binary = np.zeros_like(distance)
            binary[distance <= max_dist] = 1

            # compute number of connections between backbone atoms {"N", "CA", "C"]
            bb_connections = sum(sum(binary[0:3, 0:3]))

            # determine whether residue 'i' or residue 'j' has more atoms
            # based on this we compute number the backbone-sidechain connections
            max_num_atoms = np.argmax([len(ati), len(atj)])
            bc_connections = sum(sum(binary[:, :3])) if max_num_atoms == 1 else sum(sum(binary[:3, :]))

            # apply filter to binary for masking backbone connections
            bb_filter = np.zeros(shape=(3, 3))
            binary[:3, :3] = bb_filter              # assuming backbone atoms come first

            # compute number of connections between sidechains
            sc_connections = sum(sum(binary))

            aa_contacts[f"{aai}_{aaj}"] = [bb_connections, bc_connections, sc_connections]
            stat_contacts.append([bb_connections, bc_connections, sc_connections])

            # increase 'j-th' counter
            ctr_j += atj_pos_per_res.shape[0]

        # increase 'i-th'
        ctr_i += ati_pos_per_res.shape[0]

    return aa_contacts, np.asarray(stat_contacts)

def sampling(input_data: np.ndarray):

    # define output container
    adj, edge, feat, heads, embeddings, labels, masks, skipped = [], [], [], [], [], [], [], []

    try:

        # setting up PCA for dimensionality reduction
        # n_components = min(8, 256)
        # pca = PCA(n_components=n_components)

        # unpack input data
        graph_header, graph_atoms, graph_positions = input_data["h"], input_data["a"], input_data["p"]
        graph_residue_ids, graph_residues = input_data["i"], input_data["r"]
        graph_sse_ids, graph_sse = input_data["s"], input_data["e"]
        graph_embeddings, graph_labels = input_data["q"], input_data["l"]

        print(f"unpacked data..")

        # get motif information for portential masking
        palmscan_file = "../files/ps2_rdrp_short_motifs.csv"
        palm_df = pd.read_csv(palmscan_file, sep=",", header=0)
        motif_names = palm_df["header"].tolist()
        motif_asta, motif_aend = palm_df["start_a"].tolist(), palm_df["end_a"].tolist()
        motif_bsta, motif_bend = palm_df["start_b"].tolist(), palm_df["end_b"].tolist()
        motif_csta, motif_cend = palm_df["start_c"].tolist(), palm_df["end_c"].tolist()

        # iterate over all elements per unpacked graph sample
        for idx, (header, atoms, resids, ress, atom_pos, embs, label) in enumerate(zip(graph_header, graph_atoms,
                                                                                       graph_residue_ids, graph_residues,
                                                                                       graph_positions, graph_embeddings,
                                                                                       graph_labels)):

            label = int(label)
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

                    motif_range = mask_motif_config[mask_modifier]
                    print(f"\tset mask ranges for {header}.")

                else:
                    print(f"\t'masking' was flagged for RdRp sample, but not valid motif was found for {header} --> skip")
                    continue

            else:
                motif_range = [-1]
                print(f"\t'masking' was flagged, but couldn't find valid motif information for {header}.")

            # each sample is considered its own graph in the end
            G = nx.Graph()

            # MAPPING ATOMS TO RESIDUES
            # -------------------------------------------------------------------------------------------------------- #
            # get number of atoms per residue based on the unique residue number (ascending)
            atom_range = [np.sum(resids == rid) for rid in set(resids)]

            # reduce residues to every single occurrence as in the sequence
            res_redux = np.asarray([f"{ress[i]}:{str(resid+1)}" for i, resid in enumerate(resids) if i == 0 or resid != resids[i-1]])
            resid_redux = np.asarray([resid for i, resid in enumerate(resids) if i == 0 or resid != resids[i-1]])

            assert resid_redux.shape[0] == resid_redux.shape[0]

            # mapping start and end atoms for each residue & retrieve the corresponding position attributes
            res_atom_map = [(sum(atom_range[:i]), sum(atom_range[:i + 1])) for i in range(len(atom_range))]
            res_atom_pos = np.asarray([np.asarray(atom_pos[sta:end].astype(float)) for sta, end in res_atom_map], dtype=object)
            atom_dist_t = ARGS["distance"]

            print(f"\tcomputing atom contacts..")

            # compute residue contacts based on atom distances and whether atoms are part of backbone or sidechain structure
            # results in a dictionary where each key is an aa-pair containing a list with number of connections [bb, bc, sc]
            res_contacts, _ = compute_residue_contacts(positions=atom_pos, atom_to_aa_map=res_atom_pos,
                                                       aa_atom_keys=res_redux, max_dist=atom_dist_t)

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

            # determine masked residues
            mask = np.intersect1d(resid_redux, motif_range)

            # map SSE nodes according to their positional vectors, residues and indices
            node_map = {n: i for i, n in enumerate(res_redux)}
            pos_map = {res_redux[i]: p for i, p in enumerate(res_pos)}
            emb_map = {res_redux[i]: e for i, e in enumerate(embs)}
            mask_map = {res_redux[i]: m for i, m in enumerate(mask)}

            print(f"\tmapped SSEs")

            if 0 in mask:
                print(f"\tfound partially masked SSEs")

            # set intra-backbone connections between SSE
            # -------------------------------------------------------------------------------------------------------- #
            intra_bb_edge_pairs = [(node, res_redux[j + 1]) for j, node in enumerate(res_redux[:-1])]

            print(f"\tsetting up backbone connections..")

            for bb_i, bb_j in intra_bb_edge_pairs:

                # get positions of current connected nodes
                bbi_idx, bbj_idx = node_map[bb_i], node_map[bb_j]
                bbi_pos, bbj_pos = pos_map[bbi_idx], pos_map[bbj_idx]

                # determine edge weights & kinds
                edge_weight = 0.75 * (abs(bbi_pos - bbj_pos))

                G.add_edge(bb_i, bb_j, kind="backbone", weight=edge_weight, source=f"{bb_i}_{bb_j}", color="#ff8000")

            # add edges between SSE that are close to each other based
            # (1) on their NUMBER OF RESIDUE CONTACTS
            # -------------------------------------------------------------------------------------------------------- #
            # compute all node combinations
            pairwise_nodes = sorted(list(combinations(res_redux, 2)))

            print(f"\tsetting up atom connections..")

            for node_pair in pairwise_nodes:

                # split node pair
                resi, resj = node_pair

                # get node data per pair
                resi_pos, resj_pos = pos_map[resi], pos_map[resj]

                # atom level connections
                # ---------------------------------------------------------------------------------------------------- #
                # generate key for querying the connection table
                aa_pair = f"{resi[0]}_{resj[0]}"

                # retrieve atom distances
                bb, bc, sc = res_contacts[aa_pair] if aa_pair in res_contacts.keys() else [0, 0, 0]

                # check if there is connection between both SSE based on the pre-defined connection table
                if (aa_pair == "E_E") and (bb > 1 or bc > 2):
                    has_atom_connection = True

                elif (aa_pair == "E_H" or aa_pair == "H_E" or aa_pair == "E_T" or aa_pair == "T_E" or
                      aa_pair == "H_T" or aa_pair == "T_H") and (bb > 1 or bc > 3 or sc > 3):
                    has_atom_connection = True

                elif (aa_pair == "H_H" or aa_pair == "T_T") and (bc > 3 or sc > 3):
                    has_atom_connection = True

                else:
                    has_atom_connection = False

                # determine edge weights & kinds
                edge_weight = abs(resi_pos - resj_pos)

                # updating / adding edge with corresponding weights
                if has_atom_connection:
                    G.add_edge(resi, resj, kind="atom", weight=edge_weight, source=f"{resi}_{resj}", color="#00FF00")

            # DISCARD FILTER
            # -------------------------------------------------------------------------------------------------------- #
            num_edges = len(list(G.edges()))
            num_nodes = len(list(G.nodes()))

            # check if graph has minimum number of nodes
            if num_nodes < ARGS["min_num_nodes"]:
                print(f"ERROR\tgraph has insufficient number of nodes n={num_nodes}")
                print()
                skipped.append(header)
                continue

            # UPDATING NODES
            # -------------------------------------------------------------------------------------------------------- #
            sse_map = {"C": 0, "T": 1, "H": 2, "E": 3}

            for ids, res in enumerate(res_redux):
                G.nodes[res]["degree"] = G.degree(res)
                G.nodes[res]["emb"] = emb_map[res]
                G.nodes[res]["mask"] = mask_map[res]

            # GRAPH PLOT
            # -------------------------------------------------------------------------------------------------------- #
            print(f"\tfinalized SSE graph --> #{num_nodes} nodes | #{num_edges} edges.")

            # CRAFTING RAW FEATURES
            # -------------------------------------------------------------------------------------------------------- #
            # Node features
            node_deg = np.asarray(list(nx.get_node_attributes(G, "degree").values()))
            node_embs = np.asarray(list(nx.get_node_attributes(G, "emb").values()), dtype=object)
            node_mask = np.asarray(list(nx.get_node_attributes(G, "mask").values()), dtype=object)

            if node_deg.shape[0] != node_embs.shape[0] != node_mask.shape[0]:
                print(f"ERROR\tdimensions mismatch on raw node features\n\t{node_deg.shape[0]} != {node_embs.shape[0]}"
                      f" != {node_mask.shape[0]}")
                skipped.append(header)
                print()
                continue

            # check if node features are computed correctly
            if node_deg.shape[0] != len(list(G.nodes())):
                print(f"ERROR\tpotential mismatch between node_feats, embeddings & number of nodes!\n\t{node_deg.shape[0]} != {len(list(G.nodes()))}")
                skipped.append(header)
                print()
                continue

            print(f"\tset node features.")

            # Edge features
            edge_indices, edge_attr = get_sparse_adjacency_matrix(graph=G, weight_id="weight")

            print(f"\tdetermined adjacency.")

            if np.isnan(edge_attr).any():
                with open("../files/nan_file.txt", "a") as nan_file:
                    nan_file.write(f"{header}\n")

                print(f"ERROR\t@{header} detected NaN-values in edge attributes")
                skipped.append(header)
                print()
                continue

            print(f"\tset edge features.")

            # ATTACHMENT
            # -------------------------------------------------------------------------------------------------------- #
            adj.append(edge_indices)
            edge.append(edge_attr)
            feat.append(node_deg)
            heads.append(header)
            embeddings.append(node_embs)
            masks.append(node_mask)
            labels.append(label)

            print()

        return adj, edge, feat, heads, embeddings, masks, labels, skipped

    except Exception as e:
        traceback.print_exc()
        sys.exit(-1)


if __name__ == "__main__":

    print()

    # get user input
    ARGS = vars(flags())

    # where to find basic data
    data_dir = "/mnt/twincore/compvironas/user/nico/sse_graphs/"
    template_data = None

    # determine test and train data files
    if ARGS["test_head"] is not None:
        input_file = ARGS["inp"]
        output_file = ARGS["test_head"]

    else:
        train_data_dir = "/home/boeker/Nicolai/project/113_protein_structure_comparison/100_data/100_8_raw_train_noPDB/train_samples/"
        input_file = f"{train_data_dir}/train.npz"
        template_data = pd.read_csv("../files/templates_min.csv", sep=",", header=0)
        output_file = "raw_train.npz"

    # load data & chunking
    data = np.load(input_file, allow_pickle=True)
    num_samples = len(data["h"].tolist())

    print(f"#{num_samples} data samples loaded from numpy")
    print(f"sampling...")

    # unpack
    adj_data, edge_data, feature_data, header_data, embedding_data, mask_data, label_data, _ = sampling(input_data=data)

    # filter for templates
    to_delete = []

    print(f"... finished sampling")
    print()

    if template_data is not None:

        for (tax, head, _) in zip(template_data["order"], template_data["sample"], template_data["score"]):

            # find index of template sample
            template_idx = [idx for idx, h in enumerate(header_data) if h == str(head)]

            if len(template_idx) > 0:

                _id = template_idx[0]
                print(f"Template: {tax} ---> {head}")
                print(f"----- ID: {_id}")

                a, d, x = adj_data[_id], edge_data[_id], feature_data[_id]
                h, q, y = header_data[_id], embedding_data[_id], label_data[_id]
                m = mask_data[_id]

                # save each template individually
                np.savez(data_dir + f"raw_{tax}_template.npz",
                         a=np.array([a], dtype=int),
                         d=np.array([d], dtype=float),
                         x=np.array([x], dtype=float),
                         m=np.array([m], dtype=int),
                         h=np.array([h], dtype=str),
                         q=np.array([q], dtype=object),
                         y=np.array([y], dtype=int))

                to_delete.append(_id)

            else:
                print(f"Template: {tax} ---> {head}")
                print(f"----- ID: 'None'")

    print(f"#{len(to_delete)} assigned as templates.")

    # convert to numpy
    np_adj = np.array([[a[0], a[1]] for i, a in enumerate(adj_data) if i not in to_delete], dtype=object)
    np_edge = np.array([e for i, e in enumerate(edge_data) if i not in to_delete], dtype=object)
    np_feat = np.array([f for i, f in enumerate(feature_data) if i not in to_delete], dtype=object)
    np_head = np.array([h for i, h in enumerate(header_data) if i not in to_delete], dtype=object)
    np_emb = np.array([q for i, q in enumerate(embedding_data) if i not in to_delete], dtype=object)
    np_mask = np.array([m for i, m in enumerate(mask_data) if i not in to_delete], dtype=object)
    np_label = np.array([l for i, l in enumerate(label_data) if i not in to_delete], dtype=object)

    assert np_adj.shape[0] == np_edge.shape[0] == np_feat.shape[0] == np_emb.shape[0] \
           == np_head.shape[0] == np_label.shape[0] == np_mask.shape[0]

    data_size = np_adj.shape[0]

    if data_size == 0:

        file_name = data["h"][0]
        print(f"@{file_name} is empty..")
        print()

    else:

        # save remaining samples
        np.savez(data_dir + output_file, a=np_adj, d=np_edge, x=np_feat, h=np_head, q=np_emb, m=np_mask, y=np_label)

        print(f"processed #{np_label.shape[0]} samples")
        print(f"@class (0): #{len(np.where(np_label==0)[0])}")
        print(f"@class (1): #{len(np.where(np_label==1)[0])}")
        print()
