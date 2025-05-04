from pathlib import Path
import argparse as ap


def flags():
    parser = ap.ArgumentParser()
    parser.add_argument("-i", "--inp", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    print()

    args = flags()

    # define output directory
    out_dir = "/home/boeker/Nicolai/project/113_protein_structure_comparison/100_data/700_4_test_samples_complete_fastas/clean/"

    # get sequence that have to be cleaned
    fastas = [str(p) for p in Path(args.inp).glob("*.fa*")]

    for fasta in fastas:

        with open(fasta, "r") as src:
            data = src.read().splitlines()

        header, seq = data[0], data[1]
        current_seq_len = len(seq)

        if "X" in seq:
            seq = seq.replace("X", "")
            new_seq_len = len(seq)

            assert new_seq_len < current_seq_len

        with open(f"{out_dir}{header[1:]}.fa", "w") as tar:
            tar.write(f"{header}\n{seq}\n")

        print(f"cleand {header[1:]}.\n")
