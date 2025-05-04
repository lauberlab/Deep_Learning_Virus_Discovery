import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import random
import numpy as np
import argparse as ap
from torch_geometric.loader import DataLoader
import warnings
from tester import Tester

from dataset.graph_set import GraphSamplingSet
from trainer_classifier import Trainer
from utils import (save_with_pickle, create_target_dir, k_fold_train_test_split, model_init, get_checkpoint, )
from plots import (train_plot, test_plot)
from configs import config
from configs import paths

warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=10_000)


def select_parameter_config(p: int):
    if p == 0:
        return config.parameter_autoencoder

    elif p == 1:
        return config.parameter_classifier_01

    elif p == 2:
        return config.parameter_classifier_02

    elif p == 3:
        return config.parameter_classifier_03

    elif p == 4:
        return config.parameter_classifier_04

    else:
        sys.exit(-1)


def setting_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def display(step: str):
    print(f"\n{step}")
    print(f"#" * 25)


def get_arguments():
    parser = ap.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str, required=True)
    parser.add_argument('--retrain', type=str, default=None,
                        help="put model 'tag' you want to retrain here.")
    parser.add_argument('-r', "--rank", type=int, default=None)
    parser.add_argument('-c', "--config", type=int, default=1, choices={0, 1, 2, 3, 4})
    parser.add_argument('-p', '--profile', type=str, default="ProxyAnchorTriplet")
    parser.add_argument('-op', '--only_positive', action="store_true", default=False)
    parser.add_argument('-d', '--device', type=str, default=None)

    return parser.parse_args()


# ---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    # get user arguments
    ARGS = get_arguments()

    # load configs
    params = select_parameter_config(ARGS.config)
    params["final_mlp"] = True

    # Seeding
    seed_value = 1330
    setting_seed(seed_value)

    params["seed_value"] = seed_value

    # Set GPU if available
    params["device"] = params["device"] if ARGS.device is None else ARGS.device
    print(f"\nComputation device: {params['device']}")
    print(f"model tag: {ARGS.tag}")

    # Creating output folders for various artifacts while training
    # ---------------------------------------------------------------------------------------------------------------- #
    target_directory = f"artifacts/classifier/{ARGS.tag}/"
    create_target_dir(target_directory)

    checkpoint_directory = f"{target_directory}checkpoints/"
    create_target_dir(checkpoint_directory)

    train_directory = f"{target_directory}train/"
    create_target_dir(train_directory)

    template_directory = f"{target_directory}templates/"
    create_target_dir(template_directory)

    results_directory = f"{target_directory}results/"
    create_target_dir(results_directory)

    # TRAINING
    # ---------------------------------------------------------------------------------------------------------------- #
    display("Dataset Creation")

    # load data
    base_data = np.load(paths.train_data, allow_pickle=True)
    base_set = GraphSamplingSet(data=base_data, set_type="train", motif_masking=params["motif_masking"], training=True,
                                only_positive_class=ARGS.only_positive)

    # Determine input dimensions
    # ---------------------------------------------------------------------------------------------------------------- #
    base_sample = base_set.data[0]

    # update params input dimensions
    params['x_dim'] = base_sample.x.shape[1]
    params['e_dim'] = base_sample.edge_attr.shape[-1]
    params['num_nodes'] = base_set.get_num_nodes()
    params['num_edges'] = base_set.get_num_edges()

    # print and save params
    # ---------------------------------------------------------------------------------------------------------------- #
    # Save params to disk
    save_with_pickle(params, target_directory + "params.pkl")
    print(f"saved params to [{target_directory + 'params.pkl'}].\n")

    # print general params
    print(f"epochs: {params['epochs']}")
    print(f"batch size: {params['batch_size']}")
    print(f"stop patience: {params['stop_patience']}")
    print(f"hidden channels: {params['hidden_channels']}")

    # print input sizes
    print(f"x dim: {params['x_dim']} | e dim: {params['e_dim']}")
    print(f"num nodes: {params['num_nodes']}")
    print(f"num edges: {params['num_edges']}")
    print(f"motif_masking: {params['motif_masking']}.")

    # saving test results
    results = []

    # K-Fold Cross Validation
    # ---------------------------------------------------------------------------------------------------------------- #
    if params["k_fold"] is not None:

        # get the base data sets
        # ------------------------------------------------------------------------------------------------------------ #
        train_folds, test_folds = k_fold_train_test_split(base_set.data, k=params["k_fold"], test_size=0.1,
                                                          seed_value=30, random_state=13)

        # start k-fold cross validation
        # ------------------------------------------------------------------------------------------------------------ #
        for k, (train_fold, test_fold) in enumerate(zip(train_folds, test_folds), start=1):

            # Model creation
            # for each training turn create a new model and optimizer to reset their weights
            # -------------------------------------------------------------------------------------------------------- #
            model_save = f"{checkpoint_directory}0{k}_model"  # model save tag
            model_params = params

            # init autoencoder
            model = model_init(**model_params)
            model.to(params["device"])

            # define loss function, optimizer for backprop and scheduler for early stopping
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=model_params["alpha"],
                                          weight_decay=model_params["weight_decay"])

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=model_params["alpha_reduce"],
                                                                   patience=model_params["alpha_patience"],
                                                                   verbose=True)

            # weight sampling for DataLoader ==> stratifying batches to maintain class distribution
            # -------------------------------------------------------------------------------------------------------- #
            print(f"\n[BATCHING]: generating balanced train batches")

            """
            batch stratification might lead towards model bias; every batch contains at least n positive samples using
            those as reference for the computation of the current batch loss; this helps the model to adapt to specific 
            distributions of the data within a batch, BUT the model also becomes reliant on the presence of a positive
            sample --> in exploration data the model might be prone to overestimating the likelihood of positives, 
            leading to an increase in false positives overall

            thus, we avoid stratified batching for now!
            """
            train_loader = DataLoader(dataset=train_fold, batch_size=params["batch_size"], shuffle=False)

            # test and template data
            # -------------------------------------------------------------------------------------------------------- #
            test_loader = DataLoader(dataset=test_fold, batch_size=params["batch_size"], shuffle=False)

            # Training Loop
            # ---------------------------------------------------------------------------------------------------------#
            display("Training")
            TR = Trainer(device=params["device"], criterion=criterion, optimizer=optimizer,
                         scheduler=scheduler, patience=params["stop_patience"],
                         model_save=model_save, unfreeze=params["freeze_to"])

            print(f"[FOLD 0{k}]: {len(train_fold)} training & {len(test_fold)} test samples")
            loss_stats, metric_stats = TR.train_loop(model=model,
                                                     loader=train_loader,
                                                     num_epochs=params["epochs"],
                                                     k=k)

            # plot training loss
            epochs = list(range(len(loss_stats)))
            train_plot(epochs, loss_stats, save_as=f"{train_directory}/loss_plot_0{k}.svg")

            # save loss data to disk
            save_with_pickle(loss_stats, train_directory + f"train_loss_0{k}.pickle")

            # Prepare Testing
            # -------------------------------------------------------------------------------------------------------- #
            # reset model states
            checkpoint = get_checkpoint(params["load_from_checkpoint"], target_directory, k)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.motif_masking = False
            model.to(params["device"])
            model.eval()

            print(f"[Validation] model.motif_masking set to = {model.motif_masking}.")

            # initiate encoder class
            TESTER = Tester(model=model, device=params["device"], template_dir=template_directory, criterion=criterion,
                            classify=params["classify"], results_dir=results_directory, k=k)

            test_results, test_metrics = TESTER.test_binary(test_loader=test_loader)
            test_results.to_csv(f"{results_directory}/test_0{k}.csv")
            print(f"saved #{k}th test results.")

            # Plot test results and UMAP transformation
            # -------------------------------------------------------------------------------------------------------- #
            test_plot(results=test_results, save_as=f"{results_directory}/test_plot_0{k}.svg", y_str="Logits")
            acc, prc, rec, f1 = test_metrics

            metrics = pd.DataFrame(data={"ACC": [acc], "PRC": [prc], "REC": [rec], "F1": [f1]})
            metrics.to_csv(f"{results_directory}/test_0{k}.csv")
            print(f"saved #{k}th test results.")
