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
import traceback


def get_arguments():

    # user flags
    parser = ap.ArgumentParser()
    parser.add_argument('--inp', type=str, required=True, help="possible usage of 'train_samples'")
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--use_fasta', action='store_true', default=False)
    parser.add_argument('--only_pdb', action='store_true', default=False)
    parser.add_argument('--chunk_size', default=40, choices={24, 32, 48, 64, 128}, type=int,
                        help="set chunk size for ESM Fold model.")
    parser.add_argument('--num_recycles', default=5, type=int, help="")

    return parser.parse_args()


# define execution function
def exec_esmfold(data_stream, device: str, save_to: str, node_name: str, num_recycles: int,
                 chunk_size: int, only_pdb: bool = False):

    # init ESM Fold model
    # ---------------------------------------------------------------------------------------------------------------- #
    esm_model = esm.pretrained.esmfold_v1()
    esm_model = esm_model.eval().to(device)
    esm_model.set_chunk_size(chunk_size)

    # Outputs
    # ---------------------------------------------------------------------------------------------------------------- #
    headers, atoms, resid, residues, positions, embeddings = [], [], [], [], [], []
    sse_ids, sse_sss, labels = [], [], []

    # iterating sub-samples (per sample/file ==> ORFs)
    # ---------------------------------------------------------------------------------------------------------------- #
    general_plddt = {"header": [], "plddt": []}

    for i, (head, seq, label) in enumerate(data_stream):

        # reduce head info
        head = head.split(" ")[0]

        print(f"{head}: {len(seq)}")
        with torch.no_grad():

            # filter out short samples
            if isinstance(seq, float):
                print(f"\n[SKIP] @{i}:{head} --> type mismatch.\n")
                continue

            if len(seq) < 100:
                print(f"\n[SKIP] @{i}:{head} --> size of {len(seq)}.\n")
                continue

            # generate protein fold
            current_fold = esm_model.infer(sequences=seq.upper(), num_recycles=num_recycles)

            # structural embedding
            embedding = current_fold["s_s"].cpu().numpy()
            embedding = np.squeeze(embedding, axis=0)

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
            pdb_, atoms_, resid_, residues_, positions_ = to_pdb(protein)
            print(f"[{i + 1}|{head}]: folded protein consisting of {len(atoms_)} atoms.")

            # save PDB
            pdb_file = save_to + f"/{head}.pdb"
            save_as_pdb(out_dir=save_to, out_file=pdb_file, data=pdb_)
            print(f"[{i + 1}|{head}]: saved as .pdb-format.")

            if not only_pdb:

                # append PDB data
                atoms.append(atoms_)
                resid.append(resid_)
                residues.append(residues_)
                positions.append(positions_)
                embeddings.append(embedding)
                headers.append(head)

                # define SSE information
                SSE_default = {"C": (1, "C", "Coil"), "T": (2, "T", "Turn"), "H": (3, "H", "Helix"),
                               "E": (4, "E", "Strand")}
                SSE_target_dir = "/home/boeker/Nicolai/project/113_protein_structure_comparison/100_data/100_6_dssp/raw/"
                SSE_expected_size = len(list(set(resid_)))
                SSE_Module = SSEModule(target_dir=SSE_target_dir, SSE_default=SSE_default, expected_size=SSE_expected_size, enforce=True)

                # get DSSP readout
                dssp_file = SSE_Module.DSSP_run(input_file=pdb_file, header=head)
                dssp_data = SSE_Module.DSSP_read(file=dssp_file)

                if dssp_data is None:
                    print(f"[SSE NOT FOUND] skipping @{head} due to missing DSSP data.")
                    continue

                # get STRIDE readout
                stride_file = SSE_Module.STRIDE_run(input_file=pdb_file, header=head)
                stride_data = SSE_Module.STRIDE_read(file=stride_file)

                # get consensus
                sse_data, has_size = SSE_Module.consensus(ss1=dssp_data, ss2=stride_data)

                if not has_size:
                    print(f"[SSE-MISMATCH|SKIP] found inconsistent SSE sizes @{head}")
                    continue

                # consensus on SSE structure
                sse_id_, sse_ss_ = np.asarray([dd[0] for dd in sse_data]), np.asarray([dd[1] for dd in sse_data])

                # append DSSP data
                sse_ids.append(sse_id_)
                sse_sss.append(sse_ss_)
                labels.append(label)

                # plddt score
                struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
                general_plddt["plddt"].append(struct.b_factor.mean())
                general_plddt["header"].append(head)

                # delete tmp files, e.g. DSSP and STRIDE
                os.system(f"rm {dssp_file}")
                os.system(f"rm {stride_file}")

    assert len(atoms) == len(resid) == len(residues) == len(positions) == len(sse_ids) == len(sse_sss)

    if len(atoms) > 0:

        print(f"[PLDDT-SCORE] {sum(general_plddt['plddt']) / len(general_plddt['plddt'])}.")

        plddt_df = pd.DataFrame(data=general_plddt)
        plddt_df.to_csv(f"{save_to}/{node_name}_plddt.csv", index=None, header=True)

        # Save output data if not per sample (e.g. for training)
        # ------------------------------------------------------------------------------------------------------------ #
        numpy_file = save_to + f"/{node_name}"

        save_numpy = {
            "head": headers, "atoms": atoms, "resid": resid, "residues": residues, "positions": positions,
            "ssid": sse_ids, "ss": sse_sss, "embeddings": embeddings, "label": labels
        }

        # save main data
        save_as_numpy(data=save_numpy, file_name=numpy_file)


def parallel_execute(files: list, save_to: str, to_device: str, num_recycles: int, chunk_size: int,
                     use_fasta: bool, only_pdb: bool):

    # iterating files (=samples)
    for idx, file in enumerate(files):

        file_name = file.split("/")[-1].split(".")[0]
        node_name = file_name + ".npz"                                                      # used for the .npz output

        # Input sequences
        # ------------------------------------------------------------------------------------------------------------ #
        if not use_fasta:
            inp_data = pd.read_csv(file, sep=",", header=None)
            inp_headers, inp_seqs, inp_labels = inp_data[0], inp_data[1], inp_data[3]

        else:
            with open(file, "r") as src_file:
                content = src_file.read().splitlines()

            inp_headers, inp_seqs = [content[0][1:]], ["".join(content[1:])]
            inp_labels = [0] * len(inp_headers)

        if len(inp_headers) != len(inp_seqs) != len(inp_labels):
            print("[FATAL ERROR]: mismatch between data columns.!")
            sys.exit(-1)

        else:
            print(f"[INFO]: {len(inp_seqs)}# sub-samples @{node_name[:-4]}.\n")

        # Execute ESMFold
        assumed_output = save_to + node_name
        if os.path.exists(assumed_output):
            print(f"[SKIP]: found existing output @{assumed_output}")
            continue

        else:
            exec_esmfold(data_stream=zip(inp_headers, inp_seqs, inp_labels), device=to_device,
                         save_to=save_to, node_name=node_name, num_recycles=num_recycles, chunk_size=chunk_size,
                         only_pdb=only_pdb)


if __name__ == "__main__":

    args = get_arguments()

    # set timer
    overall_running_time_START = time.time()

    # try:

    # determine whether we need parallel GPU processing
    if args.inp == "train_samples":

        # where to find train samples
        # dir_name = "/home/boeker/Nicolai/project/113_protein_structure_comparison/" \
        #           "004_RNA_replicase_similarity_v2/DGTMAE_RNARepDT/files/"

        dir_name = "/home/boeker/Nicolai/project/113_protein_structure_comparison/004_RNA_replicase_similarity_v3/DGTMAE_RNARepDT/files/"

        # create target output dir
        output_directory = args.out + f"train_samples/"
        create_target_dir(output_directory)

        # read-in data from .csv
        data = pd.read_csv(dir_name + "train.csv", sep=",", header=None)
        exec_esmfold(data_stream=zip(data[0], data[1], data[3]), device="cuda:0", save_to=output_directory,
                     node_name="train", num_recycles=args.num_recycles,
                     chunk_size=args.chunk_size)

        overall_running_time_END = time.time()
        print(f"[FIN] running time of {(overall_running_time_END - overall_running_time_START) / 60} min.")

    else:

        file_type = "*.fa*" if args.use_fasta else "*.csv"

        # collect all .csv files from input directory
        dir_name = args.inp.split("/")[-2]
        orfs_to_process = [str(p) for p in Path(args.inp).rglob(file_type)]

        # Create target output dir
        output_directory = args.out

        # determine index for splitting lists and loading onto all available GPUs
        # split_index = len(orfs_to_process) // 2

        # Initialize the multiprocessing context
        set_start_method('spawn')

        # split lists
        # otp_1, otp_2 = orfs_to_process[:split_index], orfs_to_process[split_index:]
        # print(f"[SPLIT]: {len(otp_1)}# files on GPU_1;\n         {len(otp_2)}# files on GPU_2.")

        # split processes
        proc1 = Process(target=parallel_execute, args=(orfs_to_process, output_directory, 'cuda:0', args.num_recycles,
                                                       args.chunk_size, args.use_fasta, args.only_pdb))

        # proc2 = Process(target=parallel_execute, args=(otp_2, output_directory, 'cuda:1',
        #                                               args.keep_pdb, args.num_recycles, args.chunk_size))

        # Start the processes
        proc1.start()
        # proc2.start()

        # Wait for the processes to finish
        proc1.join()
        # proc2.join()

        overall_running_time_END = time.time()
        print(f"[FIN] running time of {(overall_running_time_END - overall_running_time_START) / 60} min.")

    # except Exception as e:
    #    print(f"[ERROR]: {e}\n")
    #    traceback.print_exc()
