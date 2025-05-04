import logging
import warnings
import numpy as np
import argparse as ap
import pandas as pd
import torch
import torch.nn as nn
import random
import traceback
import os
from torch.utils.data import DataLoader
from pathlib import Path
from tester import Tester
from dataset.graph_set import GraphDataSet, transform_data, collate_fn
from utils import (load_from_pickle, create_target_dir, model_init, load_model_params)
import configs.paths as PATHS
from configs.config import reference_header as REFERENCES
from postprocessing import postprocessing
from PML.pml_profiles import get_pml_profile
from torch.multiprocessing import Process, set_start_method, Value
import time


def setting_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_arguments():

    parser = ap.ArgumentParser()

    # required flags
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--cpus', type=int, required=True)
    parser.add_argument('--model_tag', type=str, required=True)
    parser.add_argument('--model_rank', type=int, required=True)
    parser.add_argument('--infer', type=str, required=True)
    parser.add_argument('--references', type=str, required=True, choices={"res1", "res2"})

    # optional flags
    parser.add_argument('--t', type=float, default=0.0)
    parser.add_argument('--metacontigs', action="store_true", default=False)
    parser.add_argument('--server', action="store_true", default=False)
    parser.add_argument('--batchsize', type=int, default=None)

    return parser.parse_args()


def infer(files_to_scan: list, params_per_file, checkpoints, gpu_id: int, cpus_per_gpu: int):

    # LIMIT CPU USAGE
    torch.set_num_threads(cpus_per_gpu)

    # DETERMINE GPU
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # SEEDING
    seed_value = params_per_file["seed_value"]
    setting_seed(seed_value)

    # DEFINE GLOBAL OUTPUT
    global_results = {"header": [], "origin": [],"score": [], "group": [], "group_len": [], "target": [],
                      "length": [], "hmml": [], "hmmr": []}

    # ################################################################################################################ #
    # ################################################################################################################ #
    # SET-UP MODEL
    model = model_init(**params_per_file)

    model.load_state_dict(checkpoints['model_state_dict'])
    model.motif_masking = params_per_file["motif_masking"]
    model.to(device)
    model.eval()

    logging.info(f"load_from_checkpoint={params_per_file['load_from_checkpoint']}")

    # ################################################################################################################ #
    # ################################################################################################################ #

    # extract HMM hits from provided data sheet
    hmm_hits = "./files/NIDO_RdRp_domains.csv"
    hmm_data = pd.read_csv(hmm_hits, header=0, sep=",")

    # ################################################################################################################ #
    # ################################################################################################################ #
    if params_per_file["final_mlp"]:

        y_type = "Logits"
        criterion = nn.BCEWithLogitsLoss()
        tester = Tester(model=model, device=device, criterion=criterion, classify=params_per_file["final_mlp"])

    else:

        """
        GET THE RAW TEMPLATE GRAPH AND COMPUTE THE EMBEDDING AGAIN to ensure it has the same floating point operation
        incorporated as the test embeddings; since it is just loaded, it might differ when the model was trained on
        another system with deviating floating point operations;
        """

        # LOAD TEMPLATE IN CASE: SETTING METRIC LEARNING
        template = f"template_0{params_per_file['model_rank']}.pkl"
        template_embeddings = load_from_pickle(f"{params_per_file['model_dir']}/templates/" + template)
        logging.info(f"using training template @{template}")

        # determine embedding size
        embedding_size = params_per_file["hidden_channels"] * 2 if params_per_file["pooling"] == "set" \
            else params_per_file["hidden_channels"]

        # LOAD CORRECT METRIC LEARNING PROFILE
        criterion, _, y_type, _ = get_pml_profile(profile=params_per_file["profile"],
                                                  loss_margin=params_per_file["loss_margin"],
                                                  embedding_size=embedding_size,
                                                  norm_embeddings=True)

        logging.info(f"type={y_type}")

        # INITIATE TESTER INSTANCE
        tester = Tester(model=model, device=device)

    # ################################################################################################################ #
    # ################################################################################################################ #
    # INFERENCE
    logging.info(f"start inference..\n")

    with (open(params_per_file['model_dir'] + f"/error_log.txt", "w") as err_log):

        for idx, file in enumerate(files_to_scan):

            try:
                # single file name
                split_path = file.split("/")
                file_header = split_path[-1][:-4]

                logging.info(f"{idx + 1} - {file_header}:")

                # generate dataset
                data = np.load(file, allow_pickle=True)

                data_dict, reference_dict, num_nodes = transform_data(
                    data=data, name=file_header,
                    num_min_neighbours=params_per_file["num_min_neighbours"],
                    neighbour_fraction=params_per_file["neighbour_fraction"],
                    distance_threshold=params_per_file["distance_threshold"],
                )

                dataset = GraphDataSet(data_dict=data_dict)

                # get origin
                origin_labels = np.unique(dataset.y)[0]

                # get dataset size
                len_dataset = dataset.x.shape[0]
                logging.info(f"\tdataset size: {len_dataset}")

                if len_dataset == 0:
                    logging.info(f"\tskip --> no data samples in {file_header}\n")
                    continue

                data_loader = DataLoader(dataset=dataset, batch_size=params_per_file["batch_size"], shuffle=False,
                                         collate_fn=collate_fn)

                # #################################################################################################### #
                # check for either:
                # SETTING: BINARY CLASSIFICATION
                # ---------------------------------------------------------------------------------------------------- #
                if params_per_file["final_mlp"]:

                    results, metrics = tester.test_binary(data_loader)
                    logging.info(f"\tcomputing logits")

                # SETTING: METRIC LEARNING
                # ---------------------------------------------------------------------------------------------------- #
                else:
                    references = REFERENCES[params_per_file["references"]]
                    results, ref, ref_ctr = tester.test_similarity(test_loader=data_loader,
                                                                   template=template_embeddings,
                                                                   reference=references)

                    logging.info(f"\tcomputing {y_type} matrix")

                # #################################################################################################### #
                # #################################################################################################### #
                # UNPACK RESULTS
                distances = results["y"]
                samples = results["samples"]

                # CHECK FOR SHORT INPUT (consisting only of 1 snippet) ---> sequence length ~ 100
                if len_dataset == 1:
                    samples = np.asarray(list(range(1, 2))).astype(int)
                    distances = np.asarray(distances)

                assert distances.shape[0] == samples.shape[0]

                # OVERWRITE
                results["y"] = distances
                results["samples"] = samples

                # SET PARAMS
                local_params = {"file": file_header,
                                "y_type": y_type,
                                "save_to": params_per_file["output_dir"],
                                "metacontigs": params_per_file["metacontigs"],
                                "t": params_per_file["t"]
                                }

                # #################################################################################################### #
                # POSTPROCESSING
                logging.info(f"\tstart postprocessing..")

                try:
                    pp_return = postprocessing(results=results,
                                               hmm_data=hmm_data,
                                               params=local_params)

                except Exception as e:
                    error_str = f"[PP-ERROR @{idx}|{file}]: {traceback.print_exc()}\n"
                    err_log.write(error_str)
                    continue

                # unpack postprocessing return
                local_sample, local_score, local_group, hl, hr = pp_return
                group_len = len(local_group) if len(local_group) > 0 else -1
                target = sum(local_group) // group_len

                # save and plot local screen
                # ---------------------------------------------------------------------------------------------------- #
                # save local screen results in csv format
                local_results = pd.DataFrame(data=results)
                local_results.to_csv(f"{params_per_file['output_dir']}/{file_header}_results_0{gpu_id}.csv")

                # Append to global screening results
                # ---------------------------------------------------------------------------------------------------- #
                global_results["header"].append(file_header)
                global_results["origin"].append(origin_labels)
                global_results["score"].append(float(local_score))
                global_results["group"].append(tuple(local_group))
                global_results["group_len"].append(group_len)
                global_results["target"].append(target)
                global_results["length"].append(len_dataset)
                global_results["hmml"].append(hl)
                global_results["hmmr"].append(hr)

                logging.info(f"\tsaved local & global results.")
                print()

                # #################################################################################################### #
                # #################################################################################################### #

            except Exception as e:
                print()
                logging.error(f"@sample {file_header} ==> {e}")
                traceback.print_exc()

    # plot global results
    # ---------------------------------------------------------------------------------------------------------------- #
    # finalize plot
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        global_df = pd.DataFrame(data=global_results)
        global_df.to_csv(f"{params_per_file['output_dir']}/00{gpu_id}_{params_per_file['infer']}.csv")


if __name__ == "__main__":

    print()

    # Initialize the multiprocessing context
    set_start_method('spawn')
    overall_running_time_START = time.time()
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    # gather user input
    arguments = get_arguments()

    # define path to where to find the model
    model_dir = f"./artifacts/classifier/{arguments.model_tag}"

    # create output folders
    output_dir = f"{model_dir}/inferring/" if not arguments.server else PATHS.output_infer
    output_subdir = output_dir + f"{arguments.infer}"

    create_target_dir(output_dir)
    create_target_dir(output_subdir)

    # load model params from disk + enriching with more information
    load_params, checkpoint_classifier = load_model_params(model_dir, arguments.model_rank)
    # load_params["profile"] = arguments.profile
    load_params["motif_masking"] = False
    load_params["model_dir"] = model_dir
    load_params["output_dir"] = output_subdir
    load_params["metacontigs"] = arguments.metacontigs
    load_params["infer"] = arguments.infer
    load_params["t"] = arguments.t
    load_params["references"] = arguments.references
    logging.info(f"@params:\n\n{load_params}\n")

    if arguments.batchsize is not None:
        load_params["batch_size"] = arguments.batchsize

    # global paths
    base_data_dir = PATHS.base_data_infer if arguments.server else PATHS.base_data_infer_local
    exploration_data_dir = base_data_dir + arguments.infer + "/"

    # collect data files
    data_files = [str(path) for path in Path(exploration_data_dir).glob("*.npz")]
    random.shuffle(data_files)

    logging.info(f"collected #{len(data_files)} data files")

    # split files evenly among all available GPUs
    G, N = arguments.gpus, len(data_files)
    P = int(N / G) if N % G == 0 or G == 1 else int(N / G) + 1

    # construct partial lists to distribute on each CPU
    partial_files = [data_files[i:i+P] for i in range(0, N, P)]
    num_processes = len(partial_files)
    partial_params = [load_params] * num_processes
    partial_checkpoints = [checkpoint_classifier] * num_processes

    # number of CPUs per process
    C = int(arguments.cpus / num_processes)

    assert num_processes <= G

    print(f"Num processes: {num_processes}")

    gpu_processes = []
    for i in range(num_processes):

        p = Process(target=infer, args=(partial_files[i], partial_params[i], partial_checkpoints[i], i, C))
        gpu_processes.append(p)
        p.start()

    for gp in gpu_processes:
        gp.join()

    overall_running_time_END = time.time()
    print(f"[FIN] running time of {(overall_running_time_END - overall_running_time_START) / 60} min.")