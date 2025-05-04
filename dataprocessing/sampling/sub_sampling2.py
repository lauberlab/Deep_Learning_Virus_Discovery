import time

import numpy as np
import datetime
import argparse as ap
from pathlib import Path
import subprocess as sp
import os
from graph_sampler import sampling
import multiprocessing as mp


def get_arguments():

    parser = ap.ArgumentParser()
    parser.add_argument("--cores", "-c", type=int, default=12)
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--calpha", action="store_true", default=False)
    parser.add_argument("--compressed", action="store_true", default=False)
    parser.add_argument("--distance", type=int, default=4, help="distance threshold between residues.")
    parser.add_argument("--min_num_nodes", type=int, default=9, help="number of kernels used.")
    parser.add_argument("--masking", action="store_true", default=False, help="masking motifs.")
    parser.add_argument("--cluster", action="store_true", default=False, help="masking motifs.")

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


def subsampling_from_array(inp_data: tuple, counter: int, shift: int, win_size: int, tag: str,
                           distance_threshold: int, calpha: bool, min_num_nodes: int):

    # output data container
    pos_out, feat_out, head_out, label_out = [], [], [], []

    # unpack
    smpl_atoms, smpl_resids, smpl_res, smpl_pos, smpl_sse_ids, smpl_sse, smpl_embs, smpl_label, smpl_head = inp_data

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

        # GENERATING SNIPPETS
        # multiple occurrences of the same element due to the nature of protein structure mechanisms
        # converted indexing needed
        snp_atoms = smpl_atoms[sta:end]
        snp_resids = smpl_resids[sta:end]
        snp_res = smpl_res[sta:end]
        snp_pos = smpl_pos[sta:end, :]

        # single occurrences hence standard indexing is sufficient
        snp_dssp_ids = smpl_sse_ids[idx:idx+win_size]
        snp_dssp_ss = smpl_sse[idx:idx+win_size]
        snp_embs = smpl_embs[idx:idx+win_size]

        # label and header
        snp_label, snp_head = smpl_label, smpl_head + f"_0{counter}"

        # increase counter
        counter += 1

        # one snippet is regarded as one complete graph; thus, we convert each snippet into a SSE graph
        graph_data = [snp_head, snp_atoms, snp_pos, snp_resids, snp_res, snp_dssp_ids, snp_dssp_ss, snp_embs, snp_label]

        # execute graph sampling
        feat, pos, header, label = sampling(
            input_data=graph_data, tag=tag, distance_threshold=distance_threshold,
            calpha=calpha, min_num_nodes=min_num_nodes,
        )

        # appendix
        if feat is not None:
            feat_out.append(feat)
            pos_out.append(pos)
            head_out.append(header)
            label_out.append(label)

    return [feat_out, pos_out, head_out, label_out], counter


def process_worker(files_to_process: list):

    for fi, n_file in enumerate(files_to_process):

        # file name
        file_tag = n_file.split("/")[-1][:-4]

        # define output file
        output_file = output_directory + f"/{file_tag}.npz"

        # check if output already exists
        if os.path.exists(output_file):
            print(f"found @{file_tag}")
            continue

        try:

            # open numpy file
            data = np.load(n_file, allow_pickle=True)

            # summarize data to samples
            samples = zip(data["a"], data["i"], data["r"], data["p"], data["s"], data["e"], data["q"], data["l"],
                          data["h"])

            # init counter for tagging header
            counter_data = 1

            # iterate over all samples per file
            pos_save, feat_save, head_save, label_save = [], [], [], []

            print(f"@{file_tag}: found  {data['h'].shape[0]}# pre-samples..")
            for i, sample in enumerate(samples):
                return_data, counter_data = subsampling_from_array(
                    inp_data=sample, counter=counter_data, shift=1, win_size=150, distance_threshold=args.distance,
                    calpha=args.calpha, min_num_nodes=args.min_num_nodes, tag=file_tag,
                )

                # combine samples
                feat_save.extend(return_data[0])
                pos_save.extend(return_data[1])
                head_save.extend(return_data[2])
                label_save.extend(return_data[3])

            assert len(feat_save) == len(pos_save) == len(head_save) == len(label_save)

            data_size = len(feat_save)
            print(f"@{file_tag}: processed #{data_size} samples")

            if data_size == 0:

                print(f"@{file_tag} is empty..")
                print()

            else:

                # save remaining samples
                np.savez(output_file,
                         p=np.array(pos_save, dtype=object),
                         x=np.array(feat_save, dtype=object),
                         h=np.array(head_save, dtype=object),
                         y=np.array(label_save, dtype=object))

        except Exception as err:

            current_date = datetime.date.today().strftime('%Y-%m-%d')
            error_message = f"{current_date} - Error {fi}: {file_tag} --> {err}\n"
            print(error_message)


if __name__ == "__main__":

    print()

    # log file
    log_file = f"error_logs.txt"

    # read-in user params
    args = get_arguments()

    # derive base directory name
    node_name = args.input.split("/")[-2]

    # define target output directory
    output_directory = args.output
    create_target_dir(output_directory)

    # collect all files for sub-sampling
    file_extension = "*.zst" if args.compressed else "*.npz"
    numpy_files = sorted([str(p) for p in Path(args.input).rglob(file_extension)])

    # split files evenly among all available CPUs
    C, N = args.cores, len(numpy_files)
    P = int(N / C) + 1 if N % C != 0 else int(N / C)

    # construct partial lists to distribute on each CPU
    partial_files = [numpy_files[i:i+P] for i in range(0, N, P)]
    num_processes = len(partial_files)

    assert num_processes <= C

    shared_cluster = mp.Value('b', args.cluster)
    partial_cluster = [shared_cluster] * num_processes

    """
    # start and join multiple processes
    with mp.Pool(processes=num_processes) as pool:
        pool.map(process_worker, zip(partial_files, partial_cluster))

    """
    processes = []
    for i in range(num_processes):

            p = mp.Process(target=process_worker, args=(partial_files[i], ))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()
