import numpy as np
import datetime
import argparse as ap
from pathlib import Path
import subprocess as sp
import os


def get_arguments():

    parser = ap.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)

    return parser.parse_args()


def create_target_dir(path: str, head: str = ""):

    if not os.path.exists(path):

        cmd = ["mkdir", path]
        process = sp.Popen(cmd)
        process.wait()

        print(f"[{head}|DIR]: created target dir @{path}.")


def save_as_numpy(in_data: dict, file_name: str):

    # store output in numpy format
    np.savez(file_name,
             h=np.array(in_data["head"], dtype=object),                                        # head
             a=np.array(in_data["atoms"], dtype=object),                                       # atoms
             i=np.array(in_data["resid"], dtype=object),                                       # residue index
             r=np.array(in_data["residues"], dtype=object),                                    # residues
             p=np.array(in_data["positions"], dtype=object),                                   # coordinates
             s=np.array(in_data["ssid"], dtype=object),                                        # SSE IDs
             e=np.array(in_data["ss"], dtype=object),                                          # SSE abbreviations
             q=np.array(in_data["embeddings"], dtype=object),                                  # sequence embedding
             l=np.array(in_data["label"], dtype=object),                                       # label
             )


def consecutive_resids(unique_resids: np.array):

    is_consecutive = True

    start = unique_resids[0]
    for idx in unique_resids[1:]:

        if idx == start + 1:
            start = idx

        else:
            is_consecutive = False
            break

    return is_consecutive


def subsampling_from_array(inc_data: tuple, counter: int, shift: int, win_size: int):

    # temporary data container
    tmp_atoms, tmp_resids, tmp_residues, tmp_positions, tmp_dssp_ids, tmp_dssp_ss, tmp_emb = [], [], [], [], [], [], []
    tmp_label, tmp_head = [], []

    # unpack
    smpl_atoms, smpl_resids, smpl_res, smpl_pos, smpl_sse_ids, smpl_sse, smpl_embs, smpl_label, smpl_head = inc_data

    # define definite start point
    temp_start = sorted(list(set(smpl_resids)))[0]

    # curate 'resids' (check for integrity)
    unique_residue_ids, residue_id_counts = np.unique(smpl_resids, return_counts=True)
    is_curated = consecutive_resids(unique_resids=unique_residue_ids)

    if not is_curated or temp_start != 0:
        smpl_resids = np.concatenate([np.repeat(i, c) for i, c in zip(list(range(unique_residue_ids.shape[0])), residue_id_counts)])
        print(f"[{smpl_head}] reset 'residue counter'.")

    # find definite end point
    definite_end = sorted(list(set(smpl_resids)))[-1]

    for idx in range(0, max(definite_end - win_size + 1, 1), shift):

        sta = np.where(smpl_resids == idx)[0][0]

        window_end = (idx + win_size) - 1
        end_ = window_end if window_end <= definite_end else definite_end
        end = np.where(smpl_resids == end_)[0][-1]

        # multiple occurrences of the same element due to the nature of protein structure mechanisms
        # converted indexing needed
        tmp_atoms.append(smpl_atoms[sta:end])
        tmp_resids.append(smpl_resids[sta:end])
        tmp_residues.append(smpl_res[sta:end])
        tmp_positions.append(smpl_pos[sta:end, :])

        # single occurrences hence standard indexing is sufficient
        tmp_dssp_ids.append(smpl_sse_ids[idx:idx+win_size])
        tmp_dssp_ss.append(smpl_sse[idx:idx+win_size])
        tmp_emb.append(smpl_embs[idx:idx+win_size])

        # single element
        tmp_label.append(smpl_label)
        tmp_head.append(smpl_head + f"_0{counter}")
        counter += 1

    return tmp_atoms, tmp_resids, tmp_residues, tmp_positions, tmp_dssp_ids, tmp_dssp_ss, tmp_emb, tmp_label, tmp_head, counter


if __name__ == "__main__":

    print()

    # log file
    log_file = f"./logs/error_logs.txt"

    # read-in user params
    args = get_arguments()

    # derive base directory name
    node_name = args.input.split("/")[-2]

    # define target output directory
    output_directory = args.output
    create_target_dir(output_directory)

    # collect all files for sub-sampling
    numpy_files = sorted([str(p) for p in Path(args.input).glob("*.npz")])

    # iterate over all files
    with open(log_file, "a") as log:

        for fi, n_file in enumerate(numpy_files):

            # file name
            file_tag = n_file.split("/")[-1][:-4]

            # output file
            output_file = output_directory + f"/{file_tag}"

            # outputs
            headers, atoms, resid, residues, positions, embeddings = [], [], [], [], [], []
            sse_ids, sse_ss, labels = [], [], []

            if os.path.exists(output_file):
                print(f"found @{file_tag}")
                continue

            try:

                # open numpy file
                data = np.load(n_file, allow_pickle=True)

                # summarize data to samples
                samples = zip(data["a"], data["i"], data["r"], data["p"], data["s"], data["e"], data["q"], data["l"], data["h"])

                # init counter for tagging header
                data_counter = 1

                # iterate over all samples per file
                for i, sample in enumerate(samples):

                    post_atoms, post_resid, post_residues, post_pos, post_sse_id, post_sse_ss, post_emb, post_label,\
                        post_head, post_ctr = subsampling_from_array(inc_data=sample, counter=data_counter, shift=1,
                                                                     win_size=150)

                    print(f"[{i + 1}]: #{len(post_atoms)} sub-samples with {sum([pa.shape[0] for pa in post_atoms])} atoms.")

                    # update data counter
                    data_counter = post_ctr

                    # pack sub-samples
                    atoms.extend(post_atoms)
                    resid.extend(post_resid)
                    residues.extend(post_residues)
                    positions.extend(post_pos)
                    sse_ids.extend(post_sse_id)
                    sse_ss.extend(post_sse_ss)
                    embeddings.extend(post_emb)
                    labels.extend(post_label)
                    headers.extend(post_head)

                assert len(atoms) == len(resid) == len(residues) == len(positions) == len(sse_ids) \
                       == len(sse_ss) == len(embeddings) == len(headers) == len(labels)

                if len(atoms) > 0:

                    print(f"\n[{n_file}] #{len(atoms)} overall samples.")

                    # Save output data if not per sample (e.g. for training)
                    # ------------------------------------------------------------------------------------------------ #
                    save_numpy = {
                        "head": headers, "atoms": atoms, "resid": resid, "residues": residues, "positions": positions,
                        "ssid": sse_ids, "ss": sse_ss, "embeddings": embeddings, "label": labels
                    }

                    save_as_numpy(in_data=save_numpy, file_name=output_file)

            except Exception as err:

                current_date = datetime.date.today().strftime('%Y-%m-%d')
                error_message = f"{current_date} - Error {fi}: {file_tag} --> {err}\n"
                log.write(error_message)
                print(error_message)
