import os
import argparse as ap
import pandas as pd
import numpy as np
import esm
import torch
from torch.multiprocessing import Process, set_start_method
import time
from protein import Protein, to_pdb
from sse import SSEModule
from pathlib import Path
import sys
import biotite.structure.io as bsio
from fold_utils import save_as_pdb, create_target_dir, atom14_to_atom37, save_as_numpy
import logging


def get_arguments():

    # user flags
    parser = ap.ArgumentParser()
    parser.add_argument('--inp', type=str, required=True, help="use 'train' for predicting train sample folds.")
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--keep_pdb', action='store_true', default=False)
    parser.add_argument('--chunk_size', default=40, choices={24, 32, 48, 64, 128}, type=int,
                        help="set chunk size for ESM Fold model.")
    parser.add_argument('--num_recycles', default=5, type=int, help="")
    parser.add_argument('--log_file', default=None, type=str, help="")

    return parser.parse_args()


# define execution function
def exec_esmfold(data_stream, device: str, save_to: str, node_name: str, keep_pdb: bool, num_recycles: int,
                 chunk_size: int):

    # init ESM Fold model
    # ---------------------------------------------------------------------------------------------------------------- #
    esm_model = esm.pretrained.esmfold_v1()
    esm_model = esm_model.eval().to(device)
    esm_model.set_chunk_size(chunk_size)

    # Outputs
    # ---------------------------------------------------------------------------------------------------------------- #
    headers, atoms, resid, residues, positions, embeddings = [], [], [], [], [], []
    sse_ids, sse_ss, labels = [], [], []

    # iterating sub-samples (per sample/file ==> ORFs)
    # ---------------------------------------------------------------------------------------------------------------- #
    # tmp data
    pdb_files, plddt_over_all = [], []

    data_counter = 1
    for i, (head, seq, label) in enumerate(data_stream):

        with torch.no_grad():

            # sample size restrictions & type mismatches, e.g. float
            if len(seq) < 100 or len(seq) > 600:
                logging.info(f"skip: @{i}:{head} --> size of {len(seq)}")
                continue

            if type(seq) == float:
                logging.info(f"skip: @{i}:{head} --> type mismatch")
                continue

            # generate protein fold
            current_fold = esm_model.infer(sequences=seq.upper(), num_recycles=num_recycles)

            # structural embedding
            embedding = current_fold["s_s"].cpu().numpy()
            pre_embedding = np.squeeze(embedding, axis=0).astype(np.float32)

            # extracting atom positions first and cast output to CPU
            final_atom_positions = atom14_to_atom37(current_fold["positions"][-1], current_fold)
            final_atom_positions = final_atom_positions.cpu().numpy()
            current_fold = {k: v.to("cpu").numpy() for k, v in current_fold.items()}

            # defining a mask to reduce dimensions on attributes & converting it into a boolean mask
            final_atom_mask = current_fold["atom37_atom_exists"]

            # defining a protein
            protein = Protein(
                aatype=current_fold["aatype"][0],
                atom_positions=final_atom_positions[0],
                atom_mask=final_atom_mask[0],
                residue_index=current_fold["residue_index"][0],
                b_factors=current_fold["plddt"][0],
                chain_index=current_fold["chain_index"][0] if "chain_index" in current_fold else None,
            )

            # get PDB string and attributes in Numpy format
            pdb_data, pre_atoms, pre_resid, pre_residues, pre_pos = to_pdb(protein)
            logging.info(f"{i + 1}|{head}: folded.")

            # gather and save PDB
            pdb_file = save_to + f"/{head}.pdb"
            save_as_pdb(out_dir=save_to, out_file=pdb_file, data=pdb_data)
            pdb_files.append(pdb_file)
            logging.info(f"{i + 1}|{head}: saved as .pdb-format.")

            # define SSE information
            SSE_default = {"C": (1, "C", "Coil"), "T": (2, "T", "Turn"), "H": (3, "H", "Helix"),
                           "E": (4, "E", "Strand")}
            SSE_target_dir = "/home/boeker/Nicolai/project/113_protein_structure_comparison/100_data/100_6_dssp/raw/"
            SSE_expected_size = len(list(set(pre_resid)))
            SSE_Module = SSEModule(target_dir=SSE_target_dir, SSE_default=SSE_default,
                                   expected_size=SSE_expected_size,  enforce=True)

            # get DSSP readout
            dssp_file = SSE_Module.DSSP_run(input_file=pdb_file, header=head)
            dssp_data = SSE_Module.DSSP_read(file=dssp_file)

            if dssp_data is None:
                logging.info(f"skip: due to missing DSSP data @{head}.")
                continue

            # get STRIDE readout
            stride_file = SSE_Module.STRIDE_run(input_file=pdb_file, header=head)
            stride_data = SSE_Module.STRIDE_read(file=stride_file)

            # get consensus
            sse_data, has_size = SSE_Module.consensus(ss1=dssp_data, ss2=stride_data)

            if not has_size:
                logging.info(f"skip: found inconsistent SSE sizes @{head}.")
                continue

            pre_sse_id, pre_sse_ss = np.asarray([dd[0] for dd in sse_data]), np.asarray([dd[1] for dd in sse_data])

            # pack data
            atoms.append(pre_atoms)
            resid.append(pre_resid)
            residues.append(pre_residues)
            positions.append(pre_pos)
            sse_ids.append(pre_sse_id)
            sse_ss.append(pre_sse_ss)
            embeddings.append(pre_embedding)
            labels.append(label)
            headers.append(head)

            # plddt score
            struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
            plddt_over_all.append(struct.b_factor.mean())

            # delete tmp files, e.g. DSSP and STRIDE
            os.system(f"rm {dssp_file}")
            os.system(f"rm {stride_file}")

    assert len(atoms) == len(resid) == len(residues) == len(positions) == len(sse_ids) == len(sse_ss)

    if len(atoms) > 0:

        logging.info(f"@{node_name} --> #{len(atoms)} samples.")
        logging.info(f"@{node_name} --> plddt: {sum(plddt_over_all) / len(plddt_over_all)}.\n")

        # delete temporary single PDB files
        if not keep_pdb:

            for pdb in pdb_files:
                os.system(f"rm {pdb}")

        # Save output data if not per sample (e.g. for training)
        # ------------------------------------------------------------------------------------------------------------ #
        # file_name = args.file_name if args.file_name else "complete"
        numpy_file = save_to + f"/{node_name}"

        save_numpy = {
            "head": headers, "atoms": atoms, "resid": resid, "residues": residues, "positions": positions,
            "ssid": sse_ids, "ss": sse_ss, "embeddings": embeddings, "label": labels
        }

        save_as_numpy(data=save_numpy, file_name=numpy_file)


def parallel_execute(files: list, save_to: str, to_device: str, keep_pdb: bool, num_recycles: int,
                     chunk_size: int, logger: list):

    # iterating files (=samples)
    for idx, file in enumerate(files):

        file_split = file.split("/")
        node_name = file_split[-1][:-4] + ".npz"                                              # used for the .npz output

        # Input sequences
        # ------------------------------------------------------------------------------------------------------------ #
        inp_data = pd.read_csv(file, sep=",", header=0)
        inp_headers, inp_seqs, inp_labels = inp_data["header"], inp_data["seqs"], inp_data["labels"]

        if len(inp_headers) != len(inp_seqs) != len(inp_labels):
            logging.error("error: mismatch between data columns.!")
            sys.exit(-1)

        else:
            logging.info(f"{len(inp_seqs)}# pre-samples @{node_name[:-4]}.\n")

        # Execute ESMFold
        assumed_output = save_to + node_name

        if os.path.exists(assumed_output):
            logging.info(f"skip: found existing output @{assumed_output}")
            continue

        elif len(logger) > 0 and node_name in logger:
            logging.info(f"skip: found existing output @'log_info'")
            continue

        else:
            exec_esmfold(data_stream=zip(inp_headers, inp_seqs, inp_labels), device=to_device,
                         save_to=save_to, node_name=node_name, keep_pdb=keep_pdb,
                         num_recycles=num_recycles, chunk_size=chunk_size)


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    print()

    # get user flags
    args = get_arguments()

    # set timer
    overall_running_time_START = time.time()

    # determine whether we need parallel GPU processing
    if args.inp == "train":

        # where to find train samples
        file_dir = "/home/boeker/Nicolai/project/113_protein_structure_comparison/100_data/700_1_train_samples_csv/pos/"

        # create target output dir
        output_directory = args.out + f"train_samples/"
        create_target_dir(output_directory)

    else:

        # collect all .csv files from input directory
        file_dir = args.inp
        dir_name = args.inp.split("/")[-2]

        # Create target output dir
        output_directory = args.out + f"{dir_name}/"
        create_target_dir(output_directory)

    # collect files to process
    data_files = [str(p) for p in Path(file_dir).rglob("*.csv")]

    # determine index for splitting lists and loading onto all available GPUs
    split_index = len(data_files) // 2

    # Initialize the multiprocessing context
    set_start_method('spawn')

    # split lists
    data_1, data_2 = data_files[:split_index], data_files[split_index:]
    logging.info(f"split: {len(data_1)}# files on GPU_1 (cuda:0) | {len(data_2)} # files on GPU_2 (cuda:1)")
    time.sleep(3)

    if args.log_file is not None:
        with open(args.log_file, "r") as log_file:
            log_info = log_file.read().splitlines()

    else:
        log_info = []

    # split processes
    proc1 = Process(target=parallel_execute, args=(data_1, output_directory, 'cuda:0', args.keep_pdb,
                                                   args.num_recycles, args.chunk_size, log_info))
    proc2 = Process(target=parallel_execute, args=(data_2, output_directory, 'cuda:1', args.keep_pdb,
                                                   args.num_recycles, args.chunk_size, log_info))

    # Start the processes
    proc1.start()
    proc2.start()

    # Wait for the processes to finish
    proc1.join()
    proc2.join()

    overall_running_time_END = time.time()
    logging.info(f"running time of {(overall_running_time_END - overall_running_time_START) / 60} min.")
    print()
