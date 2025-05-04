import sys
import numpy as np
import itertools
import traceback
from sse_utils import (compute_residue_contacts, calpha_coordinates, mean_coordinates, )


np.random.seed(1310)


def sampling(input_data: list, tag: str, distance_threshold: int = 4, calpha: bool = False,
             min_num_nodes: int = 9):

    try:

        # unpack input data
        header, atoms, atom_pos, resids, ress, ssids, sses, embs, label = input_data
        header, label = str(header), int(label)

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

        # print(f"\tcomputing atom contacts..")

        # compute residue contacts based on atom distances and whether atoms are part of backbone or sidechain structure
        # results in a dictionary where each key is an aa-pair containing a list with number of connections [bb, bc, sc]
        # res_contacts, _ = compute_residue_contacts(positions=atom_pos, atom_to_aa_map=res_atom_pos,
        #                                           aa_atom_keys=res_redux, max_dist=distance_threshold)

        # compute centroids per residue based on the mean coordinates of the atoms it comprises
        if calpha:
            res_pos = np.asarray([calpha_coordinates(pos) for i, pos in enumerate(res_atom_pos)])

        else:
            res_pos = np.asarray([mean_coordinates(pos, reduced=True) for i, pos in enumerate(res_atom_pos)])

        # check if attribute reduction was executed correctly; the number of centroids must be equal to the reduced
        # number of residues (sequence) and further match the amount of SSE elements given by DSSP
        if res_pos.shape[0] != res_redux.shape[0]:
            print(f"[{tag}|ERROR]: number of residues != number of centroids; mapping of atoms per residue failed!")
            return None

        elif res_pos.shape[0] != ssids.shape[0] != len(sses):
            print(f"[{tag}|ERROR]: number of centroids != number of SSE elements; incorrect mapping!")
            return None

        # MAPPING RESIDUES TO SSE
        # -------------------------------------------------------------------------------------------------------- #
        # reduce the SSE elements by counting their consecutive occurrence in 'sses'
        sse_redux = [(k + str(i + 1), int(len(list(g)))) for i, (k, g) in enumerate(itertools.groupby(sses))]

        # extract SSE attributes
        sse_ss, res_range = np.asarray(list(zip(*sse_redux))[0]), np.asarray(list(zip(*sse_redux))[1])

        if sum(res_range) != res_redux.shape[0]:
            print(f"[{tag}|ERROR]: residue number mismatch => {sum(res_range)} != {res_redux.shape[0]}!")
            return None

        # map residues to SSE similar to atoms per residue
        sse_res_map = [(sum(res_range[:i]), sum(res_range[:i + 1])) for i in range(len(res_range))]
        sse_res_pos = [res_pos[sta:end] for sta, end in sse_res_map]

        # compute the SSE centroids
        sse_pos = np.asarray([mean_coordinates(pos) for i, pos in enumerate(sse_res_pos)])

        if sse_pos.shape[0] != sse_ss.shape[0]:
            print(f"[{tag}|ERROR]: mismatching number of SSE!")
            return None

        # PROCESSING SSE
        # -------------------------------------------------------------------------------------------------------- #
        sse_pos, sse_nodes, sse_res, sse_embs, aa_idx, discard = [], [], [], [], 0, 0

        # convert into SSE form
        for sid, sse in enumerate(sse_ss):

            # get number of residues of current SSE element
            num_residues = res_range[sid]

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

            # append
            sse_res.append(current_res_rex)
            sse_nodes.append(sse)

            # get embeddings and sum up the embedding to a single vector of size (1, 1024)
            """
            summation of the SSE embeddings might result in information loss! instead we proceed with aa embeddings
            per SSE as node features (M, 1024) where M correlates to the number of aa belonging to the respective SSE
            """
            sse_embs.append(embs[aa_idx:aa_ide])

            # compute p and p & append to variable holding the positional space vectors
            if N > 1:
                sse_pos.append(mean_coordinates(sse_pos_range, as_array=True))

            else:
                sse_pos.append(sse_pos_range[0])

            # increase counters for aa residues (index)
            aa_idx += num_residues

        # convert to numpy
        features = np.array(sse_embs, dtype=object)
        positions = np.array(sse_pos, dtype=object)

        # discard filter
        if features.shape[0] < min_num_nodes:
            print(f"ERROR\tinsufficient number of nodes\n\t{features.shape[0]}")
            return None

        # print(f"\tset node features.")

        return features, positions, header, label

    except Exception as e:
        traceback.print_exc()
        sys.exit(-1)
