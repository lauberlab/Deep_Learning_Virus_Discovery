import esm
import torch
import argparse as ap
import numpy as np
from scipy.special import softmax
import biotite.structure.io as bsio
import gc


def get_flags():

    parser = ap.ArgumentParser()
    parser.add_argument("-f", "--fasta", type=str, required=True)
    parser.add_argument("-o", "--out", type=str, required=True)
    parser.add_argument("-c", "--chunk_size", default=40, choices={24, 32, 48, 64, 128}, type=int,
                        help="set chunk size for ESM Fold model.")
    parser.add_argument("-n", "--num_recycles", default=5, type=int)

    return parser.parse_args()


def parse_output(output):

    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0,:,1]

    bins = np.append(0,np.linspace(2.3125,21.6875,63))
    sm_contacts = softmax(output["distogram_logits"],-1)[0]
    sm_contacts = sm_contacts[...,bins<8].sum(-1)
    xyz = output["positions"][-1, 0, :, 1]
    mask = output["atom37_atom_exists"][0, :, 1] == 1
    o = {"pae": pae[mask, :][:, mask],
         "plddt": plddt[mask],
         "sm_contacts": sm_contacts[mask, ][:, mask],
         "xyz": xyz[mask]}

    return o


if __name__ == "__main__":

    args = get_flags()

    # add dir
    out_dir = args.out if args.out[-1] == "/" else args.out + "/"

    # set device initially to 'cpu'
    device = "cpu"

    """
    get all fasta entries, provided, that the fasta is curated
    """

    with open(args.fasta, "r") as fasta:
        fasta_content = fasta.read().splitlines()

    # identify header lines within the fasta file
    header_idx = [i for i, c in enumerate(fasta_content) if c[0] == ">"]
    header_idx.append(len(fasta_content))

    fasta_sequences, fasta_header = [], []
    for j, hidx in enumerate(header_idx[:-1]):

        # reduce to sub content including the header and the corresponding nucleotide sequence
        # the latter needs to be concatenated since it is split among several lines
        content = fasta_content[hidx:header_idx[j + 1]]
        fasta_header.append(content[0][1:])
        fasta_sequences.append("".join(content[1:]))

    # clean-up sequences
    clean_fasta_sequences = [seq.replace("U", "X").replace("O", "X").replace("Z", "X").replace("-", "").upper()
                             for seq in fasta_sequences]

    assert len(fasta_header) == len(clean_fasta_sequences)

    # free memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = "cuda:0"

    # load ESMmodel
    esm_model = esm.pretrained.esmfold_v1()
    esm_model = esm_model.eval().to(device)
    esm_model.set_chunk_size(args.chunk_size)

    for i, (head, seq) in enumerate(zip(fasta_header, clean_fasta_sequences)):

        print(f"{i+1} -- head: {head}\n{seq}")

        # get structure predictions
        model_output = esm_model.infer(seq, num_recycles=args.num_recycles, residue_index_offset=512)

        # convert into readable PDB-format
        pdb_string = esm_model.output_to_pdb(model_output)[0]

        # save PDB-file
        pdb_file = out_dir + f"{head}.pdb"

        with open(pdb_file, "w") as target:
            target.write(pdb_string)

        struct = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
        plddt = struct.b_factor

        plddt_all, plddt_mean = "\n".join([f"{i+1}: {p}" for i, p in enumerate(plddt)]), plddt.mean()
        plddt_string = f"{plddt_all}\n\nmean:{plddt_mean}"
        plddt_out = plddt_string.splitlines()

        # save model outputs in txt-format
        np.savetxt(out_dir + f"{head}_plddt.txt", plddt_out, fmt="%s")

        # print(f"finished @{head[1:]}: plddt --- {plddt:.3f} | ptm --- {ptm:.3f}")
        print(f"finished @{head}..")

