import argparse as ap
from sampler import sampling, cutting
from pathlib import Path
import pandas as pd
import os.path
import subprocess as sp


def flags():

    parser = ap.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sizes", type=int, default=150, help="sample sizes")
    parser.add_argument("--shift", type=int, default=None, help="sample shift")

    return parser.parse_args()


if __name__ == "__main__":

    r"""
    Test samples are derived from 3 base classes
    (1) 'Nidovirales' polyproteins (as positive class)
    (2) 'Protonido' Nidovirales ancestors (as positive class)
    (3) RNA-dependent DNA polymerases ('RNAdDNA',first negative class)
    (4) DNA-dependent DNA polymerases ('DNAdDNA', second negative class)
    """

    # get & extract user flags
    args = flags()

    inp_dir = args.inp
    out_dir = args.out
    sample_name = out_dir.split("/")[-2]

    # get test files
    # all test classes have their base name (abbr.) within their directory path
    test_files = [str(p) for p in Path(inp_dir).rglob("*.fa*")]
    print(f"found #{len(test_files)} @{inp_dir.split('/')[-2]}")

    # create target directory
    if not os.path.exists(out_dir):

        cmd = ["mkdir", out_dir]
        process = sp.Popen(cmd)
        process.wait()

        print(f"[{sample_name}|DIR]: created target dir @{out_dir}.")

    # iterate over all files
    for file in test_files:

        # read file content
        with open(file, "r") as src:
            content = src.read().splitlines()

        # extract content
        head, seq = content[0][1:], content[1]
        head = head.split(" ")[0]

        overall_seq_len = len(seq)

        # run sampler or cut sequences
        if args.shift is None:
            win_size = 500
            shift = win_size - 150

        else:
            shift = args.shift
            win_size = args.sizes

        out_dict = sampling(head=head, seq=seq, shift=shift, win_size=win_size)

        out_frame = pd.DataFrame(data=out_dict)
        out_frame.to_csv(f"{out_dir}/{head}.csv", sep=",", index=None, header=None)
        print(f"sampled {head}.")
