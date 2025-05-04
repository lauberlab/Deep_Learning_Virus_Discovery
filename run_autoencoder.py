import os
import torch
import random
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch, Dataset

import warnings
import logging
from random import Random

from model.autoencoder import SSEGraphEncoderModel
from dataset.graph_set import GraphSamplingSet
from trainer_autoencoder import AutoencoderTrainer
from utils import save_with_pickle
from configs import config

warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=10_000)


def setting_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_data(in_data: list):

    for i, data in enumerate(in_data):

        print(f"Graph {i}:")
        print(f"\tnode: {data.x.shape}")
        print(f"\tedge: {data.edge_attr.shape}")
        print(f"\teigenvec: {data.eigenvec.shape}")
        print(f"\temb: {data.emb.shape}")
        print(f"\tmask: {data.mask.shape}")
        print(f"\tadj: {data.edge_index.shape}")
        print(f"\thead: {data.head.shape}")
        print(f"\tlabel: {data.y.shape}")


def create_target_dir(path):

    if not os.path.exists(path):
        os.mkdir(path)


def batching(dataset: list, bs: int):

    def __index_sampling(smpl, used_idx, num_smpl):
        available_indices = set(range(len(smpl))) - used_idx
        sampled_indices = np.asarray(list(available_indices), dtype=int)
        return sampled_indices[:num_smpl].tolist()

    # track used indices
    used_indices = set()

    # batches
    batches = []

    while len(used_indices) < len(dataset):

        # set up a randomizer
        rng = Random(seed_value)
        indices = __index_sampling(dataset, used_indices, bs)

        # update used indices
        used_indices.update(indices)

        # sampling
        samples = [dataset[i] for i in indices]
        rng.shuffle(samples)

        batch = Batch.from_data_list(samples)
        batches.append(batch)

    return batches, len(batches)


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info("start..")

    # set parameter
    params = config.parameter_autoencoder

    # seeding
    seed_value = 1330
    setting_seed(seed_value)

    g = torch.Generator()
    g.manual_seed(seed_value)
    params["seed_value"] = seed_value

    # data preparation
    train_data = np.load(config.train_data, allow_pickle=True)
    train_set = GraphSamplingSet(data=train_data, set_type="train", only_positive_class=True,
                                 masking=params["masking"], l_dim=params["laplacian_dim"])
    train_loader = DataLoader(dataset=train_set.data, batch_size=params["batch_size"], generator=g, shuffle=False)

    # check_data(train_set.data)

    # determine data dimensions
    train_sample = train_set.data[0]

    params["x_dim"] = train_sample.x.shape[-1]
    params["e_dim"] = train_sample.edge_attr.shape[-1]
    params["q_dim"] = train_sample.emb.shape[-1]
    params["l_dim"] = train_sample.eigenvec.shape[-1]
    params["m_dim"] = train_sample.mask.shape[-1]

    logging.info(f"x_dim: {params['x_dim']} | e_dim: {params['e_dim']} | "
                 f"q_dim: {params['q_dim']} | l_dim: {params['laplacian_dim']}")

    # set up target directories for saving params, model-checkpoints etc.
    target_directory = f"./artifacts/autoencoder/{params['tag']}/"
    create_target_dir(target_directory)

    checkpoint_directory = f"{target_directory}checkpoints/"
    create_target_dir(checkpoint_directory)

    loss_directory = f"{target_directory}losses/"
    create_target_dir(loss_directory)

    # save user parameter
    save_with_pickle(params, target_directory + "params.pkl")

    # set up training environment
    device = params["device"]

    # define model save location
    model_save = f"{checkpoint_directory}at"

    # initiate model
    model = SSEGraphEncoderModel(
        seed=seed_value, x_dim=params["x_dim"], e_dim=params["e_dim"], q_dim=params["q_dim"],
        l_dim=params["l_dim"], hidden_channels=params["hidden_channels"], num_attn_heads=params["attention_heads"],
        aggregation=params["aggregation"], aggr_concat=params["aggregation_concat"], omit_edge_attr=params["omit_edge_attr"],
        attention_root=params["attention_root"], num_layers_enc=params["num_enc_layers"], pooling=params["pooling"],
        masking=params["masking"]).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=params["alpha"], weight_decay=params["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=params["alpha_reduce"],
                                                           patience=params["alpha_patience"], verbose=True)

    TR = AutoencoderTrainer(device=device, optimizer=optim, scheduler=scheduler,
                            patience=params["stop_patience"], save_loc=model_save)

    loss_stats = TR.train_loop(model=model, tr_loader=train_loader, num_epochs=params["epochs"])

    save_with_pickle(loss_stats, loss_directory + f"tr_auto_loss.pickle")
