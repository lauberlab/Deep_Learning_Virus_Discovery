import numpy as np
import argparse as ap
from pathlib import Path
import logging
import traceback


def get_arguments():

    parser = ap.ArgumentParser()
    parser.add_argument('-i', '--inp', default=None, required=True)
    parser.add_argument('-o', '--out', default=None, required=True)

    return parser.parse_args()


def merge_npz_files(filenames: list, output: str):

    # define a new output container holding all merged data
    all_data = {}

    """
    # init lists, assuming all files have the same keys
    with np.load(filenames[0], allow_pickle=True) as data:

        for key in list(data.keys()):
            all_data[key] = []

    print(list(all_data.keys()))
    logging.info(f"initialized target dict.")
    """

    # iterate over all files
    for filename in filenames:

        try:
            # read the file content as numpy
            with np.load(filename, allow_pickle=True) as data:

                # loop through each key-value pair in the loaded data, where each key represents a column and value the
                # corresponding numpy array
                for key, array in data.items():

                    # all_data[key].append(array)

                    # in case the column is already merged into the output container, concatenate the numpy array to the
                    # existing arrays in that column
                    if key in all_data:
                        current_array = all_data[key]
                        new_array = current_array + array.tolist()
                        all_data[key] = new_array

                    # if not, assign key-value pair
                    else:
                        all_data[key] = array.tolist()

            logging.info(f"added @{filename}..")

        except Exception as e:
            print(traceback.print_exc())

    print()
    print()

    # convert to numpy
    for key, array_list in all_data.items():

        for i, array in enumerate(array_list):

            array_list[i] = np.asarray(array)

        all_data[key] = np.array(array_list, dtype=object)

        logging.info(f"converted {key} to numpy array of type=object")

    # keep column tags and save merged data to disk
    np.savez(output, **all_data)


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    print()

    # get user flags
    args = get_arguments()

    logging.info(f"start merging numpy files..")

    files_to_merge = [str(p) for p in Path(args.inp).glob("*.npz")]
    output_filename = f"{args.out}train.npz"

    merge_npz_files(filenames=files_to_merge, output=output_filename)
