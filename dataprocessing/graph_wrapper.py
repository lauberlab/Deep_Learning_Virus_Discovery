import os.path
from pathlib import Path
import subprocess as sp
import argparse as ap
import multiprocessing


def flags():
    parser = ap.ArgumentParser()
    parser.add_argument("--inp", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--cpu", type=int, default=12)
    parser.add_argument("--sse", action="store_true", default=False)
    parser.add_argument("--use_calpha", action="store_true", default=False)
    parser.add_argument("--compressed", action="store_true", default=False)
    parser.add_argument("--redux_dim", type=int, default=128)
    parser.add_argument("--redux", type=str, default=None)

    return parser.parse_args()


def process_file(file_chunk: str):

    base_dir = "/mnt/twincore/ceph/cvir/user/nico/sse_graphs/"
    out_dir = base_dir if args.out is None else args.out

    for in_file in file_chunk:

        _file_ext = -4 if not args.compressed else -8
        header = in_file.split("/")[-1][:_file_ext].replace(".", "_")
        expected_output = base_dir + header

        if os.path.exists(expected_output):
            print(f"{expected_output} already exists..")
            continue

        # start subprocess
        command = ["python", "graph.py", "--inp", in_file, "--head", header, "--out", out_dir]

        if args.use_calpha:
            command += ["--calpha"]

        if args.compressed:
            command += ["--compressed"]

        if args.sse:
            command += ["--sse"]
        
        if args.redux is not None:
            command += ["--redux", args.redux]

        command += ["--redux_dim", str(args.redux_dim)]

        result = sp.run(command, stdout=sp.PIPE, stderr=sp.PIPE)

        if result.returncode == 0:
            print(f"{result.stdout.decode()}")
            print(f"finished {header}")

        else:
            print(f"ERROR\t{result.stderr.decode()}")


def parallel_process_files(files: list, num_cpus: int):

    chunk_size = (len(files) // num_cpus) + 1
    chunks = [files[i:i+chunk_size] for i in range(0, len(files), chunk_size)]

    processes = []
    for chunk in chunks:
        process = multiprocessing.Process(target=process_file, args=(chunk, ))
        processes.append(process)
        process.start()

    for p in processes:
        p.join()


if __name__ == "__main__":

    args = flags()

    # test_dir = "/home/boeker/Nicolai/project/113_protein_structure_comparison/100_data/100_9_raw_test_noPDB/"
    test_dir, file_ext = args.inp, "*.npz" if not args.compressed else "*.npz.zst"
    test_files = [str(p) for p in Path(test_dir).rglob(file_ext)]

    print(f"found #{len(test_files)} test samples to process.")

    parallel_process_files(test_files, args.cpu)

