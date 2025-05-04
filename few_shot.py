import os
import sys

import torch
import torch.nn as nn
import random
import numpy as np
import argparse as ap
import warnings

from dataset.graph_set import GraphSamplingSet
from utils import (save_with_pickle, create_target_dir, model_init)
from plots import (train_plot, roc_plot)
from configs import config
from fs_utils import (sample_task, save_model, split_train_test)
from fs_train import FSTrainer

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
    parser.add_argument('-s', '--task_size', type=int, default=15)
    parser.add_argument('-tr', "--num_train_tasks", type=int, default=150)
    parser.add_argument('-te', "--num_test_tasks", type=int, default=100)
    parser.add_argument('-m', "--max1", type=int, default=5)
    parser.add_argument('-c', "--config", type=int, default=1, choices={0, 1, 2, 3, 4})
    parser.add_argument('-d', '--device', type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":

    # get user arguments
    ARGS = get_arguments()

    # load configs
    params = select_parameter_config(ARGS.config)

    # Seeding
    seed_value = 1330
    setting_seed(seed_value)

    g = torch.Generator()
    g.manual_seed(seed_value)

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

    results_directory = f"{target_directory}results/"
    create_target_dir(results_directory)

    # load data
    base_data = np.load(config.train_data, allow_pickle=True)
    base_set = GraphSamplingSet(data=base_data, set_type="train", masking=params["masking"],
                                l_dim=params["laplacian_dim"])
    base_labels = base_set.labels

    # Determine input dimensions
    # ---------------------------------------------------------------------------------------------------------------- #
    base_sample = base_set.data[0]

    # update params input dimensions
    params['x_dim'] = base_sample.x.shape[1]
    params['e_dim'] = base_sample.edge_attr.shape[-1]
    # params['q_dim'] = base_sample.emb.shape[-1]
    params['l_dim'] = base_sample.eigenvec.shape[-1]
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
    print(f"x dim: {params['x_dim']} | e dim: {params['e_dim']} | l dim: {params['l_dim']}")
    print(f"num nodes: {params['num_nodes']}")
    print(f"num edges: {params['num_edges']}")
    print(f"masking: {params['masking']}.")

    # split data sets
    train_set, test_set = split_train_test(base_set.data, ratio=0.1, seed=seed_value)
    labels1 = [t.y for i, t in enumerate(train_set) if i == 1]

    # Model creation
    # for each training turn create a new model and optimizer to reset their weights
    # ---------------------------------------------------------------------------------------------------------------- #
    model_save = f"{checkpoint_directory}fs_model"  # model save tag
    model_params = params

    model = model_init(**model_params)
    model.to(params["device"])

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # define optimizer for backprop and scheduler for early stopping
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_params["alpha"],
                                  weight_decay=model_params["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=model_params["alpha_reduce"],
                                                           patience=model_params["alpha_patience"],
                                                           verbose=True)

    # Training
    # ---------------------------------------------------------------------------------------------------------------- #
    # init necessary meta variables for training
    device, num_epochs, loss_t, patience = params["device"], params["epochs"], params["loss_t"], params["stop_patience"]
    n_train_tasks, n_test_tasks = ARGS.num_train_tasks, ARGS.num_test_tasks
    TRAINER, TOTAL_LOSS = FSTrainer(device=device, model=model, optim=optimizer, loss_fn=criterion), []

    # training loop
    for epoch in range(num_epochs):

        # setup running variables
        train_buffer, epoch_loss, check_los, stop_ctr = {"0": [], "1": []}, 0.0, float("inf"), 0

        for i in range(n_train_tasks):

            support_set, query_set, train_buffer = sample_task(training_data=train_set, buffer=train_buffer,
                                                               seed=seed_value, max1=ARGS.max1,
                                                               task_size=ARGS.task_size)

            # train the model
            loss, optimizer = TRAINER.meta_train(support_set, query_set)
            epoch_loss += loss.item()

        TOTAL_LOSS.append(epoch_loss)
        print(f"Epoch [{epoch + 1}|{num_epochs}] ==> tr_loss: {epoch_loss:.5f}")

        if epoch_loss < loss_t:
            print(f"Epoch [{epoch + 1}|{num_epochs}]: loss threshold undercut {epoch_loss} < {loss_t}\n")
            save_model(epoch + 1, model, optimizer, epoch_loss, model_save + "_check.pth")

        if epoch_loss < check_los:

            check_los = epoch_loss
            print(f"Epoch [{epoch + 1}|{num_epochs}]: reduced loss {check_los:.5f} --> checkpoint\n")
            save_model(epoch + 1, model, optimizer, epoch_loss, model_save + "_check.pth")

            # reset early stop counter
            stop_ctr = 0

        else:
            print(f"Epoch [{epoch + 1}|{num_epochs}| Stopper] @{stop_ctr}\n")
            stop_ctr += 1                                                               # increase early stop counter

            if stop_ctr >= patience:
                print(f"Training Stop @epoch {epoch}\n")

                break

    # plot training loss
    run_epochs = list(range(len(TOTAL_LOSS)))
    train_plot(run_epochs, TOTAL_LOSS, save_as=f"{train_directory}/loss_plot.svg")

    # save loss data to disk
    save_with_pickle(TOTAL_LOSS, train_directory + f"train_loss.pickle")

    # Testing
    # ---------------------------------------------------------------------------------------------------------------- #
    test_buffer, metrics = {"0": [], "1": []}, []
    for i in range(n_test_tasks):

        support_set, query_set, buffer = sample_task(training_data=train_set, buffer=test_buffer, seed=seed_value,
                                                     max1=ARGS.max1, task_size=ARGS.task_size)

        # test step
        precision, tpr, fpr, auc, f1 = TRAINER.meta_test(support_set, query_set)

        # plot ROC AUC
        roc_plot(tpr, fpr, auc, save_as=f"{results_directory}/auc_plot.svg")

        # append to metrics container
        metrics.append((precision, tpr, f1, auc))

    # save scores after k-fold training
    # ---------------------------------------------------------------------------------------------------------------- #
    ranking = sorted(metrics, key=lambda x: x[-1], reverse=True)
    save_with_pickle(ranking, target_directory + "ranking.pkl")
