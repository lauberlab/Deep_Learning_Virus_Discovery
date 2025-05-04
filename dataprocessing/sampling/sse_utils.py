import os
import os.path
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
import plotly.graph_objects as go
from typing import List



# SECTION: HELPER FUNCS
# -------------------------------------------------------------------------------------------------------------------- #
def seeding(seed):

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    return np.random.default_rng(seed)


def assertion(inp_list: list):

    tmp_assert = []

    for li in inp_list:

        tmp_assert.append(len(li))

    print(f"{tmp_assert}")
    return all(el == tmp_assert[0] for el in tmp_assert)


def normalize_array(arr):

    norm = np.linalg.norm(arr)

    if norm == 0:

        return arr

    else:

        return arr / norm


def create_target_dir(path: str):

    import os.path
    import subprocess as sp

    if not os.path.exists(path):

        cmd = ["mkdir", path]
        process = sp.Popen(cmd)
        process.wait()


def encode_seq_to_integer(seq: str, amino_acid_code: str, padding_size: int = None):

    # init numerical translation of amino acid code
    char_to_int = dict((char, i) for i, char in enumerate(amino_acid_code))

    # encode sequence to integer
    seq_integer = np.array([float(char_to_int[char]) for char in str(seq)],
                           dtype=np.float32)

    # apply padding if flagged
    if padding_size is not None:
        pad_vector = np.zeros(shape=padding_size)  # init zero padding vector
        pad_vector[:seq_integer.shape[0]] = seq_integer  # fill up padding vector

        seq_integer = np.copy(pad_vector)

    return seq_integer


def get_palm_motifs(file_path: str = "/home/boeker/Nicolai/project/111_palmscan/output/palm_scan_info.csv"):

    # get necessary information from PalmScan results
    df = pd.read_csv(file_path, header=None, sep=",")

    palm_header = list(df[0])
    palm_origin_seq = list(df[1])
    palm_motif_seq = list(df[3])
    palm_seq_len = list(df[4])

    # get motif start and end positions converting them to integer
    motif_start = list(df[6])
    motif_end = list(df[11])

    return palm_header, palm_origin_seq, palm_motif_seq, palm_seq_len, motif_start, motif_end


def save_mod_pdb(out_dir: str, out_file: str, mod_pdb: list, head):

    # create subdirectory in case it's not existing yet
    if not os.path.exists(out_dir):
        create_target_dir(out_dir)
        print(f"[{head}|SAVE]: created new sub directory.")

    # out_file = out_dir + f"{head}.pdb"

    with open(out_file, "w") as tar:
        tar.writelines(mod_pdb)

    print(f"[{head}|SAVE]: saved modified PDB file to disk.")


def consistent_amino_acids(aa_counter):

    aa_counter = np.unique(aa_counter).tolist()

    if aa_counter[0] != 1:
        return False

    else:
        return all(x2 - x1 == 1 for x1, x2 in zip(aa_counter[:-1], aa_counter[1:]))
        # return all(aa_counter == list(range(aa_counter[0], aa_counter[-1] + 1, 1)))


# SECTION: GRAPH NETWORK(X)
# -------------------------------------------------------------------------------------------------------------------- #
def get_distance_matrix(coordinates: np.ndarray):

    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    sqr_dist = np.sum(diff**2, axis=-1)

    return np.sqrt(sqr_dist)


# more memory efficient way to store data
def get_sparse_adjacency_matrix(graph: nx.Graph, weight_id: str = None):

    adj_mat = nx.to_scipy_sparse_array(graph, weight=weight_id, format="coo")
    src, tar = adj_mat.row, adj_mat.col
    # weights = np.asarray(adj_mat.data).astype(float)
    weights = np.asarray([[float(w) for w in item.split(":")] for item in adj_mat.data]).astype(float)
    indices = np.stack((src, tar)).astype(int)

    assert weights.shape[0] == indices.shape[-1]

    return indices, weights


def positional_alignment(v1: np.ndarray, v2: np.ndarray):

    # Ensure vectors have the same dimension
    if v1.shape != v2.shape:
        raise ValueError("[ValError] vector dim mismatch.")

    # calculate dot product and magnitudes
    dot_product = np.dot(v1, v2)
    v1_mag = np.linalg.norm(v1)
    v2_mag = np.linalg.norm(v2)

    # calculate cosine of the angle
    cos_theta = dot_product / (v1_mag * v2_mag)

    # Handle potential numerical issues for cosine values outside the range [-1, 1]
    cos_theta = np.clip(cos_theta, -1, 1)

    # Calculate angle in radians, then transform into degrees
    theta = np.arccos(cos_theta)
    degree = np.degrees(theta)

    # check vector alignmentss
    if degree in list(range(80, 100)):
        align = "orthogonal"                    # margin towards 90

    elif degree in list(range(170, 190)):
        align = "antiparallel"                  # margin towards 180

    elif degree in list(range(-10, 10)):
        align = "parallel"                      # margin towards 0

    else:
        align = "mixed"

    return theta, align


def check_vector_direction(v1: np.ndarray, v2: np.ndarray, epsilon: float = 0.65):

    # compute dot product
    vi = v1 / np.linalg.norm(v1)
    vj = v2 / np.linalg.norm(v2)

    dot_product = np.round(vi.dot(vj), decimals=4)

    if dot_product > epsilon:
        vector_direction = "parallel"

    elif dot_product < -epsilon:
        vector_direction = "antiparallel"

    else:
        vector_direction = "mixed"

    return dot_product, vector_direction


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

            # compute number of connections between side chains
            sc_connections = sum(sum(binary))

            aa_contacts[f"{aai}_{aaj}"] = [bb_connections, bc_connections, sc_connections]
            stat_contacts.append([bb_connections, bc_connections, sc_connections])

            # increase 'j-th' counter
            ctr_j += atj_pos_per_res.shape[0]

        # increase 'i-th'
        ctr_i += ati_pos_per_res.shape[0]

    return aa_contacts, np.asarray(stat_contacts)


def mask_sse(current_resid_rex, motif_range):

    res_mask_overlap = len(np.intersect1d(current_resid_rex, motif_range))
    res_mask_ratio = res_mask_overlap / len(current_resid_rex)
    # print(f"mask ratio: {res_mask_ratio}")
    return 0 if res_mask_ratio >= 0.5 else 1


# SECTION: ESM prediction matrix pooling
# -------------------------------------------------------------------------------------------------------------------- #
def normalize(dat):

    norm = np.linalg.norm(dat)
    norm_dat = dat if norm == 0 else dat / norm

    return norm_dat


def feature_aggregation(embedding: np.ndarray):
    """
    we aggregate multiple embedding reductions like summation, averaging, min/max pooling

    :param  embedding of shape (N, L, 1024) where N is the number of nodes and L the number of amino acid embeddings
            per node n
    :return: stacked numpy matrix of shape (4, N, 1024) unified over L by applying different reduction methods
    """

    sum_embs, avg_embs, max_embs = [], [], []

    for emb in embedding:

        emb = emb.astype(np.float32) if emb.dtype == np.object_ else emb

        sum_embs.append(normalize(np.sum(emb, axis=0)))
        avg_embs.append(normalize(np.mean(emb, axis=0)))
        max_embs.append(normalize(np.max(emb, axis=0)))
        # min_embs.append(np.min(emb, axis=0))

    embeddings: List[np.ndarray] = [np.array(sum_embs),
                                    np.array(avg_embs),
                                    # torch.tensor(min_embs),
                                    np.array(max_embs)]

    return np.concatenate(embeddings, axis=1)


# SECTION: DATASET CREATION
# -------------------------------------------------------------------------------------------------------------------- #
def train_test_splitter(X, y, size, seed):

    # split into training, validation and test sets using scikit-learn
    # train-test split 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, stratify=y, random_state=seed)

    return X_train, X_test, y_train, y_test


def class_distribution(labels: list):

    # create dict to count classes
    dict_heads = [f"c{i}" for i in range(0, max(labels) + 1)]

    counter = {}
    for head in dict_heads:
        counter[head] = 0

    # counting classes
    for label in labels:

        counter[f"c{label}"] += 1

    return counter


def count_classes(path: str):

    # return value
    classes = {}

    # parent directory
    root = Path(path)

    for sub_dir in root.iterdir():

        count = len(list(sub_dir.iterdir()))
        classes[sub_dir.name] = count

    return classes


def save_as_np(data: pd.DataFrame, save_to: str, file_name: str = "raw_train.npz"):

    # select features and labels
    x, y = data.iloc[:, 0:-1], data.iloc[:, -1]

    # saving complete set as numpy to disk
    np.savez(save_to + file_name,
             a=np.array(x["adj"], dtype=object),
             d=np.array(x["edge"], dtype=object),
             p=np.array(x["pos"], dtype=object),
             x=np.array(x["feats"], dtype=object),
             h=np.array(x["head"], dtype=object),
             y=np.array(y, dtype=np.int32)
             )


# SECTION: PLOTTING WITH PLOTLY
# -------------------------------------------------------------------------------------------------------------------- #
# function for creating different edge traces for the scatter plot
def create_edge_trace(graph: nx.Graph):

    xe, ye, ze = [], [], []

    pos = nx.get_node_attributes(graph, "pos")
    colors = list(nx.get_edge_attributes(graph, "color").values())
    kinds = list(nx.get_edge_attributes(graph, "kind").values())

    for node_i, node_j in graph.edges(data=False):
        xe.extend([pos[node_i][0], pos[node_j][0], None])
        ye.extend([pos[node_i][1], pos[node_j][1], None])
        ze.extend([pos[node_i][2], pos[node_j][2], None])

    edge_colors = list()
    for (col) in (colors):
        edge_colors.extend((col, col, col))

    edge_colors = np.repeat(colors, 3)
    edge_text = np.repeat(kinds, 3)

    return go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode='lines',
        line=dict(
            width=10,
            color=edge_colors),
        hoverinfo='text',
        text=edge_text,
    )


# function for creating node traces for scatter plot
def create_node_trace(graph: nx.Graph, node_size: int = 11):
    pos = nx.get_node_attributes(graph, "pos")

    xn, yn, zn = [], [], []

    for idx, (key, val) in enumerate(pos.items()):
        xn.append(val[0])
        yn.append(val[1])
        zn.append(val[2])

    return go.Scatter3d(
        x=xn, y=yn, z=zn,
        mode='markers',
        marker=dict(
            symbol="circle",
            line_width=0,
            size=node_size,
            color=["#CC00CC"] * len(xn),
        ),
        hoverinfo='text+x+y+z',
        text=list(graph.nodes()),
    )


# plotting using 'plotly'
def plotly_scatter_3d(traces: list, save_loc: str, title=""):
    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title="",
    )

    figure = go.Figure(
        data=traces,
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(t=100),
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis)
            )
        )
    )

    figure.write_html(save_loc)


# ---------------------------------------------------------------------------------------------------------------------#
# pickle saving and loading
def save_with_pickle(data, path):
    with open(path, "wb") as tar:
        pickle.dump(data, tar)


def load_from_pickle(path):
    with open(path, "rb") as src:
        return pickle.load(src)