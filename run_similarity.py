import os
import sys

import torch
import random
import numpy as np
import argparse as ap
import warnings
from torch.utils.data import DataLoader
import logging

from dataset.graph_set import GraphDataSet, transform_data, collate_fn
from trainer_similarity import Trainer
from tester import Tester
from utils import (save_with_pickle, create_target_dir, k_fold_train_test_split, model_init,
                   transform_similarity, load_model_params, get_checkpoint, load_from_pickle)
from plots import (train_plot, test_plot, roc_plot)
from PML.pml_profiles import get_pml_profile, get_miner_with_fixed_anchor
from configs import config, paths

warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=10_000)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)


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
    parser.add_argument('-s', "--server", action="store_true", default=False)
    parser.add_argument('-r', "--rank", type=int, default=None)
    parser.add_argument('-c', "--config", type=int, default=1, choices={0, 1, 2, 3, 4})
    parser.add_argument('-p', '--profile', type=str, default="ProxyAnchorTriplet")
    parser.add_argument('-op', '--only_positive', action="store_true", default=False)
    parser.add_argument('-f', '--fixed_anchor', action="store_true", default=False)
    parser.add_argument('-um', '--use_miner', action="store_true", default=False)
    parser.add_argument('-pt', '--permutation_tests', type=int, default=1)
    parser.add_argument('-d', '--device', type=str, default="cuda:0")
    parser.add_argument('-dt', '--distance_threshold', type=float, default=4)
    parser.add_argument('-rh', '--reference_header', type=str, default="res1", choices={"res1", "res2"})

    return parser.parse_args()


# ---------------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    # get user arguments
    ARGS = get_arguments()

    # load configs
    params = select_parameter_config(ARGS.config)
    params["profile"] = ARGS.profile
    params["final_mlp"] = False
    params["permutation"] = False
    params["distance_threshold"] = ARGS.distance_threshold
    P = ARGS.permutation_tests

    if P > 1:
        params["k_fold"] = 3                       # --> proxy for permutation rounds
        params["permutation"] = True

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
    # model scores
    metrics = []

    display("Dataset Creation")

    # determine the reference samples : RES1 or RES2
    reference_header = config.reference_header[ARGS.reference_header]

    # load data
    base_path = paths.train_data_cluster if ARGS.server else paths.train_data
    base_data = np.load(base_path, allow_pickle=True)

    # transform data into final graph representations
    data_dict, reference_dict, num_nodes = transform_data(
        data=base_data, name="train", rdrp_only=ARGS.only_positive, num_min_neighbours=params["num_min_neighbours"],
        neighbour_fraction=params["neighbour_fraction"], distance_threshold=params["distance_threshold"],
        references=reference_header
    )

    # custom datasets for training and as references
    base_set = GraphDataSet(data_dict=data_dict)
    logging.info(f" converted train data.")

    reference_set = GraphDataSet(data_dict=reference_dict)
    logging.info(f" converted reference data.")

    # save raw references & load for model
    save_with_pickle(reference_set, template_directory + f"raw_references.pkl")
    reference_loader = DataLoader(reference_set, batch_size=len(reference_set), shuffle=False, collate_fn=collate_fn)

    # Determine input dimensions
    # ---------------------------------------------------------------------------------------------------------------- #
    # update params input dimensions
    params['x_dim'] = base_set.x.shape[-1]
    params['num_nodes'] = num_nodes

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
    print(f"X dim: {params['x_dim']}")
    print(f"num nodes: {params['num_nodes']}")
    print(f"motif_masking: {params['motif_masking']}.")
    print(f"fixed_anchor: {ARGS.fixed_anchor}.")
    print(f"permutation testing: {params['permutation']}")

    # K-Fold Cross Validation
    # ------------------------------------------------------------------------------------------------------------ #
    if params["k_fold"] is not None:

        # Setup metric learning ==> loading PyTorch Metric Learning (PML) profiles
        # -------------------------------------------------------------------------------------------------------- #
        criterion, miner, dist_type, criterion_optim = get_pml_profile(
            profile=ARGS.profile, loss_margin=params["loss_margin"], embedding_size=params["hidden_channels"],
            norm_embeddings=True, use_miner=ARGS.use_miner)

        # get the base data sets
        # -------------------------------------------------------------------------------------------------------- #
        train_folds, test_folds = k_fold_train_test_split(base_set, p=P, k=params["k_fold"], test_size=0.1,
                                                          seed_value=30, label_permutation=params["permutation"])

        # start k-fold cross validation
        # -------------------------------------------------------------------------------------------------------- #
        for k, (train_fold, test_fold) in enumerate(zip(train_folds, test_folds), start=1):

            # Model creation
            # for each training turn create a new model and optimizer to reset their weights
            # ---------------------------------------------------------------------------------------------------- #
            model_save = f"{checkpoint_directory}0{k}_model"  # model save tag
            model_params = params

            # initiate model
            if ARGS.retrain:
                model_from = f"./artifacts/classifier/{ARGS.retrain}"
                model_params, checkpoint_classifier = load_model_params(model_from, ARGS.rank)
                assert model_params["hidden_channels"] == params["hidden_channels"]

            model = model_init(**model_params)
            model.to(params["device"])

            # define optimizer for backprop and scheduler for early stopping
            optimizer = torch.optim.AdamW(model.parameters(), lr=model_params["alpha"],
                                          weight_decay=model_params["weight_decay"])

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                   factor=model_params["alpha_reduce"],
                                                                   patience=model_params["alpha_patience"],
                                                                   verbose=True)

            # in case retrain is flagged, load model checkpoints and initiate model weights
            anchor_template = None

            if ARGS.retrain:
                model.load_state_dict(checkpoint_classifier["model_state_dict"])

                if ARGS.fixed_anchor:
                    # load template working as the anchor
                    template_string = f"{model_params['template_header']}_0{model_params['max_rank']}.pkl"
                    anchor_template = load_from_pickle(f"{model_from}/templates/" + template_string)
                    anchor_template = torch.Tensor(anchor_template).to(params["device"])

                    # overwrite miner instance to utilize fixed anchors
                    miner = get_miner_with_fixed_anchor(ARGS.profile, params["batch_size"], params["loss_margin"],
                                                        anchor_template)

            # weight sampling for DataLoader ==> stratifying batches to maintain class distribution
            # ---------------------------------------------------------------------------------------------------- #
            print(f"\n[BATCHING]: generating balanced train batches")

            """
            batch stratification might lead towards model bias; every batch contains at least n positive samples using
            those as reference for the computation of the current batch loss; this helps the model to adapt to specific 
            distributions of the data within a batch, BUT the model also becomes reliant on the presence of a positive
            sample --> in exploration data the model might be prone to overestimating the likelihood of positives, 
            leading to an increase in false positives overall
            
            thus, we avoid stratified batching for now!
            """

            train_loader = DataLoader(dataset=train_fold, batch_size=params["batch_size"],
                                      shuffle=False, collate_fn=collate_fn)

            # test and template data
            # ---------------------------------------------------------------------------------------------------- #
            test_loader = DataLoader(dataset=test_fold, batch_size=params["batch_size"],
                                     shuffle=False, collate_fn=collate_fn)

            # Training Loop
            # -----------------------------------------------------------------------------------------------------#
            display("Training")
            TR = Trainer(device=params["device"], criterion=criterion, criterion_optim=criterion_optim,
                         optimizer=optimizer, miner=miner, scheduler=scheduler, patience=params["stop_patience"],
                         model_save=model_save, unfreeze=params["freeze_to"])

            print(f"[FOLD 0{k}]: {len(train_fold)} training & {len(test_fold)} test samples")
            ref_embeddings, ref_labels, losses, _ = TR.train_loop(model=model,
                                                                  tr_loader=train_loader,
                                                                  ref_loader=reference_loader,
                                                                  num_epochs=params["epochs"], k=k,
                                                                  fixed_anchor=ARGS.fixed_anchor)

            # save template embeddings as artifacts
            save_with_pickle(ref_embeddings, template_directory + f"template_0{k}.pkl")

            # plot training loss
            epochs = list(range(len(losses)))
            train_plot(epochs, losses, save_as=f"{train_directory}/loss_plot_0{k}.svg")

            # save loss data to disk
            save_with_pickle(losses, train_directory + f"train_loss_0{k}.pickle")

            # Prepare Testing
            # ---------------------------------------------------------------------------------------------------- #
            # reset model states
            checkpoint = get_checkpoint(params["load_from_checkpoint"], target_directory, k)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.motif_masking = False
            model.to(params["device"])
            model.eval()

            print(f"[Validation] model.motif_masking set to = {model.motif_masking}.")

            # initiate encoder class
            TESTER = Tester(model=model,
                            device=params["device"],
                            results_dir=results_directory,
                            k=k)

            # run testing
            results, max_ref, ref_ctr = TESTER.test_similarity(test_loader=test_loader, template=ref_embeddings,
                                                               reference=reference_header)

            # saving results & header
            results.to_csv(f"{results_directory}/test_0{k}.csv")

            with open(f"{results_directory}max_ref_0{k}.txt", "w") as tar:
                tar.write(f"reference: {max_ref}\ncounter: {ref_ctr}")

            print(f"saved #{k}th test results.")

            # Plot test results and UMAP transformation
            # ---------------------------------------------------------------------------------------------------- #
            test_plot(results=results, save_as=f"{results_directory}/test_plot_0{k}.svg", y_str="Similarity")

            # Compute model metrics
            # ---------------------------------------------------------------------------------------------------- #
            precision, tpr, fpr, auc, f1 = transform_similarity(results)

            # plot ROC AUC
            roc_plot(tpr, fpr, auc, save_as=f"{results_directory}/auc_plot_0{k}.svg")

            # append to metrics container
            metrics.append((k, precision, tpr, (f1 + auc) / 2))

    # save scores after k-fold training
    # ---------------------------------------------------------------------------------------------------------------- #
    ranking = sorted(metrics, key=lambda x: x[-1], reverse=True)
    save_with_pickle(ranking, target_directory + "ranking.pkl")
