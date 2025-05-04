import os.path
import pandas as pd
import subprocess as sp
from pathlib import Path
import argparse as ap


# EXECUTION
if __name__ == "__main__":

    # user flags
    parser = ap.ArgumentParser()
    parser.add_argument("--inp", required=True, type=str, help="input directory.")
    parser.add_argument("--target", required=True, type=str, help="data file name.")
    args = parser.parse_args()

    # PalmScan executable
    PALM_EXEC_PATH = "/home/boeker/Nicolai/project/117_palmscan2/palmscan/bin/palmscan2"
    PALM_BASE_OUT = "/home/boeker/Nicolai/project/119_RdRp_PDB_analysis/ps2/"
    PALM_TSV_PATH = f"{PALM_BASE_OUT}tsv/"
    PALM_FASTA_PATH = f"{PALM_BASE_OUT}fa/"

    # retrieve input files
    files = [str(p) for p in Path(args.inp).rglob("*.fa*")]
    num_files = len(files)

    # declare output lists
    header, sequences, sequence_lengths, scores = [], [], [], []
    motif_sequences, motif_sequence_lengths, motif_starts, motif_ends = [], [], [], []

    # final output file
    palm_out = f"{PALM_BASE_OUT}{args.target}.csv"

    print("starting 'PalmScan2' [~]")

    for ind, file in enumerate(files):

        with open(file, "r") as src:
            content = src.read().splitlines()

        # get header and full input sequence
        head, sequence = content[0][1:], content[1]

        # define output
        out_tsv = f"{PALM_TSV_PATH}{head}.tsv"
        out_fa = f"{PALM_FASTA_PATH}{head}.fa"

        # check if temporary fasta exists
        if os.path.exists(file):

            # 1 - RUN PalmScan
            palm_cmd = [PALM_EXEC_PATH,
                        "-search_pssms", file,
                        "-tsv", out_tsv,
                        "-fasta", out_fa,
                        ]

            palm_process = sp.Popen(palm_cmd)
            palm_process.wait()

            print(f"finished 'PalmScan' for {head}.")

        # read various output files to extract further information
        # check whether a report file is generated, should be the case if the log-odds-score is > 2
        if os.path.exists(out_tsv) and os.path.exists(out_fa):

            try:
                ps_data = pd.read_csv(out_tsv, header=0, sep="\t")

            except pd.errors.EmptyDataError:
                print(f"SKIP - {head} is empty..")
                continue

            # extract columns from data matrix
            score, motif_sta, motif_end = ps_data["Score"].item(), ps_data["Lo"].item(), ps_data["Hi"].item()

            print(f"{score} - {motif_sta} - {motif_end}")

            # only keep high-confidence RdRp motifs (default from PalmScan2)
            if score >= 20.00:

                with open(out_fa, "r") as fa_src:
                    fa_data = fa_src.read().splitlines()

                # extract motif sequence containing motifs A-C
                motif_seq = "".join(fa_data[1:])

                # appendix
                header.append(head)
                sequences.append(sequence)
                sequence_lengths.append(len(sequence))  # get original sequence length
                motif_sequences.append(motif_seq)
                motif_sequence_lengths.append(len(motif_seq))
                motif_starts.append(motif_sta)
                motif_ends.append(motif_end)
                scores.append(score)

    # consolidate data
    out_dict = {"header": header, "original_seq": sequences, "original_len": sequence_lengths,
                "motif_seq": motif_sequences, "motif_len": motif_sequence_lengths,
                "motif_start": motif_starts, "motif_ends": motif_ends, "score": scores,
                }

    out_frame = pd.DataFrame(data=out_dict)
    out_frame.to_csv(palm_out, sep=",", index=None, header=list(out_dict.keys()))
    print("converted to .csv format.")
