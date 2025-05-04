from Bio import SeqIO
import numpy as np
import os.path
import subprocess as sp
from pathlib import Path
import torch
import sys

np.set_printoptions(threshold=sys.maxsize)

RESIDUE_MAP = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E", "GLN": "Q",
               "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
               "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def atom14_to_atom37(atom14, batch):
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


def tmp_fasta(sequence: str, header: str, tmp_dir: str):

    current_fasta = f"{tmp_dir}{header}.fasta"

    # create file
    cmd = ["touch", current_fasta]
    touch = sp.Popen(cmd)
    touch.wait()

    # write to file
    with open(current_fasta, "w") as tar:
        tar.write(f">{header}\n{sequence}")

    return current_fasta


def check_fastas(fasta_files: list):

    for fasta in fasta_files:

        if not os.path.exists(fasta):

            return False


def create_target_dir(path: str, head: str = ""):

    if not os.path.exists(path):

        cmd = ["mkdir", path]
        process = sp.Popen(cmd)
        process.wait()

        print(f"[{head}|DIR]: created target dir @{path}.")


def save_as_pdb(out_dir, out_file, data):

    # create subdirectory in case it's not existing yet
    if not os.path.exists(out_dir):
        create_target_dir(out_dir)

    with open(out_file, "w") as tar:
        tar.writelines(data)


def save_as_numpy(data: dict, file_name: str):

    # store output in numpy format
    np.savez(file_name,
             h=np.array(data["head"], dtype=object),                                        # head
             a=np.array(data["atoms"], dtype=object),                                       # atoms
             i=np.array(data["resid"], dtype=object),                                       # residue index
             r=np.array(data["residues"], dtype=object),                                    # residues
             p=np.array(data["positions"], dtype=object),                                   # coordinates
             s=np.array(data["ssid"], dtype=object),                                        # SSE IDs
             e=np.array(data["ss"], dtype=object),                                          # SSE abbreviations
             q=np.array(data["embeddings"], dtype=object),                                  # sequence embedding
             l=np.array(data["label"], dtype=object),                                       # label
             )


def save_as_numpy_reduced(data: dict, file_name: str):

    # store output in numpy format
    np.savez(file_name,
             h=np.array(data["head"], dtype=object),                                        # head
             i=np.array(data["resid"], dtype=object),                                       # residue index
             p=np.array(data["positions"], dtype=object),                                   # coordinates
             q=np.array(data["embeddings"], dtype=object),                                  # sequence embedding
             l=np.array(data["label"], dtype=object),                                       # label
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


def subsampling_from_array(data: list, head: str, counter: int, shift: int, win_size: int):

    # temporary data container
    tmp_atoms, tmp_resids, tmp_residues, tmp_positions, tmp_dssp_ids, tmp_dssp_ss, tmp_emb = [], [], [], [], [], [], []
    tmp_label, tmp_head = [], []

    # unpack
    atoms, resids, residues, positions, dssp_ids, dssp_ss, embedding, label = data

    # define definite start point
    temp_start = sorted(list(set(resids)))[0]

    # curate 'resids' (check for integrity)
    unique_residue_ids, residue_id_counts = np.unique(resids, return_counts=True)
    is_curated = consecutive_resids(unique_resids=unique_residue_ids)

    if not is_curated or temp_start != 0:
        resids = np.concatenate([np.repeat(i, c) for i, c in zip(list(range(unique_residue_ids.shape[0])), residue_id_counts)])
        print(f"[{head}] reset 'residue counter'.")

    # find definite end point
    definite_end = sorted(list(set(resids)))[-1]

    for idx in range(0, max(definite_end - win_size + 1, 1), shift):
        # print(f"{idx + 1}: {resids.shape[0]} --> {resids[:20]}")

        sta = np.where(resids == idx)[0][0]

        window_end = (idx + win_size) - 1
        end_ = window_end if window_end <= definite_end else definite_end
        end = np.where(resids == end_)[0][-1]

        # multiple occurrences of the same element due to the nature of protein structure mechanisms
        # converted indexing needed
        tmp_atoms.append(atoms[sta:end])
        tmp_resids.append(resids[sta:end])
        tmp_residues.append(residues[sta:end])
        tmp_positions.append(positions[sta:end, :])

        # single occurrences hence standard indexing is sufficient
        tmp_dssp_ids.append(dssp_ids[idx:idx+win_size])
        tmp_dssp_ss.append(dssp_ss[idx:idx+win_size])
        tmp_emb.append(embedding[idx:idx+win_size])

        # single element
        tmp_label.append(label)
        tmp_head.append(head + f"_0{counter}")
        counter += 1

    return tmp_atoms, tmp_resids, tmp_residues, tmp_positions, tmp_dssp_ids, tmp_dssp_ss, tmp_emb, tmp_label, tmp_head, counter


def extract_from_fastas(fasta_files: list):

    sequences, headers = [], []

    # utilizing BioPython to read-in sequences from fasta files
    for idx, fasta in enumerate(fasta_files):

        for seq_record in SeqIO.parse(fasta, "fasta"):

            sequences.append(str(seq_record.seq))
            headers.append(str(seq_record.id))

    print(f"[EXTRACT]: done.")

    return sequences, headers


def extract_from_fasta(fasta_file):

    # utilizing BioPython to read-in sequences from fasta files
    for seq_record in SeqIO.parse(fasta_file, "fasta"):

        return str(seq_record.seq), str(seq_record.id)


def slice_seq_by_shift(sequence: str, win_size: list, shift: int = 1):

    # out samples container
    sub_samples = []

    # determine final win_size
    if len(win_size) == 2 and (win_size[-1] > len(sequence) > win_size[0]):
        final_win_size = win_size[0]

    elif len(sequence) > win_size[-1]:
        final_win_size = win_size[-1]

    else:
        final_win_size = len(sequence)

    sta = 0
    end = sta + final_win_size
    while end <= len(sequence):

        sub_samples.append(sequence[sta:end])

        sta += shift
        end = sta + final_win_size

    return sub_samples

