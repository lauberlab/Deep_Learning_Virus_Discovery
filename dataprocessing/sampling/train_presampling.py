import pandas as pd
import argparse as ap
import logging

"""
- cutting all sequences (from both classes) to the same length range 
- as for the RdRp sequences they should cover more than the motifs A-C
- using PalmScan2 info about motif A to indicate the starting cut-off point
--> overlapping header (discard those RdRps we dont have a confirmed motif found)
- filter double entries based on the UniProt header/tags (especially from the negative samples)
- 
"""

def get_arguments():
    parser = ap.ArgumentParser()

    parser.add_argument('-p', '--pos', action="store_true", default=False)
    parser.add_argument('-n', '--neg', action="store_true", default=False)
    parser.add_argument('-m', '--merge', action="store_true", default=False)
    # parser.add_argument('-t', '--tolerance', type=int, default=25)
    # parser.add_argument('-tl', '--target_len', type=int, default=300)

    return parser.parse_args()


def positive_sampling():

    # PalmScan results
    palm_df = pd.read_csv("../../files/palm2_motifs.csv", sep=",")

    # attributes
    palm_headers, palm_seqs = palm_df["header"].tolist(), palm_df["origin_seq"].tolist()
    motif_starts = palm_df["motif_start"].tolist()

    assert len(palm_headers) == len(palm_seqs) == len(motif_starts)

    sequences, seq_lens, headers, labels = [], [], [], []
    for i, (header, seq, ms) in enumerate(zip(palm_headers, palm_seqs, motif_starts)):

        if not "X" in seq:

            # ss = ms - 25
            # se = ss + args.target_len
            # new_seq = seq[ss:se]

            # if len(new_seq) == args.target_len - args.tolerance:
            sequences.append(seq)
            headers.append(header)
            seq_lens.append(len(seq))
            labels.append(1)

    out_dict = {"header": headers, "seqs": sequences, "seq_lens": seq_lens,"labels": labels}

    # generate pandas dataframe and store it in csv-format
    out_frame = pd.DataFrame(data=out_dict)
    out_frame.to_csv(f"../../files/positives.csv", sep=",", index=False)

    logging.info(f"saved csv-format for positive class..")


def negative_sampling():

    # read negative samples from fasta-file
    negatives_fasta = "../../files/negative_extended.fa"
    with open(negatives_fasta, "r") as src:
        content = src.read().splitlines()

    # filter for empty lines
    content = [c for c in content if c]

    # determine header and corresponding sequences
    header_idx = [idx for idx, c in enumerate(content) if c[0] == ">"]
    header_idx.append(len(content))

    known_header, header, sequences, seq_lens, labels = [], [], [], [], []
    for i, head_idx in enumerate(header_idx[:-1]):

        # each single fasta header & sequence can be extracted by collecting all lines between 2 headers,
        # indicated by index
        single_fasta_content = content[head_idx:header_idx[i + 1]]

        # reduce to header and sequence
        fasta_header, fasta_seq = single_fasta_content[0][1:], "".join(single_fasta_content[1:])

        if "|" in fasta_header:
            fasta_header = fasta_header.split("|")[1]

        # filter out sequences that contain 'unknown' amino acid residues (not part of the standard format)
        # if "X" not in fasta_seq and len(fasta_seq) >= args.target_len and fasta_header not in known_header:
        if "X" not in fasta_seq and fasta_header not in header:

            # ss = 0
            # se = ss + args.target_len

            # head_ctr = 1
            # while se <= len(fasta_seq):
                # sub_seq = fasta_seq[ss:se]

                # if len(sub_seq) == args.target_len - args.tolerance:
            sequences.append(fasta_seq)
            seq_lens.append(len(fasta_seq))
            # header.append(f"{fasta_header}_{head_ctr}")
            header.append(fasta_header)
            labels.append(0)

                # head_ctr += 1
                # ss += args.target_len
                # se = ss +  args.target_len

        # known_header.append(fasta_header)

    out_dict = {"header": header, "seqs": sequences, "seq_lens": seq_lens, "labels": labels}

    # generate pandas dataframe and store it in csv-format
    out_frame = pd.DataFrame(data=out_dict)
    out_frame.to_csv(f"../../files/negatives.csv", sep=",", index=False)

    logging.info(f"saved csv-format for negative class..")


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    print()

    args = get_arguments()

    if args.pos:
        positive_sampling()

    if args.neg:
        negative_sampling()

    if args.merge:

        # merge & shuffle outputs
        df1 = pd.read_csv("../../files/negatives.csv")
        df2 = pd.read_csv("../../files/positives.csv")

        merged = pd.concat([df1, df2], ignore_index=True)
        shuffled = merged.sample(frac=1, random_state=3010)
        # shuffled = shuffled[merged.columns]

        shuffled.to_csv("../../files/train.csv", header=None, index=False, sep=",")

        print("merged and shuffle")

    logging.info(f"finished sampling.")
    print()