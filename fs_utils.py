import random
import numpy as np
import torch


def save_model(num_epochs: int, model, optim, loss, model_save: str):

    torch.save({'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss},
               model_save)


def split_train_test(data: list, ratio: float = 0.1, seed: float = 3010):

    rng = random.Random(seed)

    pos_samples = [sample for sample in data if sample.y == 1]
    neg_samples = [sample for sample in data if sample.y == 0]

    p_split, n_split = int(len(pos_samples) * (1 - ratio)), int(len(neg_samples) * (1 - ratio))
    train_pos_samples, test_pos_samples = pos_samples[:p_split], pos_samples[p_split:]
    train_neg_samples, test_neg_samples = neg_samples[:n_split], neg_samples[n_split:]

    # combine training set
    train_set = train_pos_samples + train_neg_samples
    rng.shuffle(train_set)

    # combine test set
    test_set = test_pos_samples + test_neg_samples
    rng.shuffle(test_set)

    return train_set, test_set


def draw_random_elements(rng, lst: list, x: int):

    return rng.sample(lst, x)


def label_indices(labels: np.ndarray, buffer: dict):

    l0_idx = [i for i, l in enumerate(labels) if l == 0 and i not in buffer["0"]]
    l1_idx = [i for i, l in enumerate(labels) if l == 1 and i not in buffer["1"]]

    return l0_idx, l1_idx


def sample_task(training_data: list, buffer: dict, seed: int = 3010, max1: int = 10, task_size: int = 20,
                test: bool = False):

    # random generator
    rng = random.Random(seed)

    if test:
        max1 = task_size

    # number of elements from positive and negative class
    num1 = rng.randint(1, max1)
    num0 = task_size - num1

    # Sampling Support Set
    # ---------------------------------------------------------------------------------------------------------------- #

    # labels
    labels = np.asarray([t.y for t in training_data])
    sl0_idx, sl1_idx = label_indices(labels, buffer)

    # draw samples based on labels
    sl0 = draw_random_elements(rng, sl0_idx, num0)
    sl1 = draw_random_elements(rng, sl1_idx, num1)

    # update buffer for support samples
    buffer["0"].extend(sl0)
    buffer["1"].extend(sl1)

    # reduce training data
    s0 = np.array(training_data)[sl0].tolist()
    s1 = np.array(training_data)[sl1].tolist()

    sample = s0 + s1
    rng.shuffle(sample)

    # Sampling Query Set
    # ---------------------------------------------------------------------------------------------------------------- #
    ql0_idx, ql1_idx = label_indices(labels, buffer)

    ql0 = draw_random_elements(rng, ql0_idx, 1)
    ql1 = draw_random_elements(rng, ql1_idx, 1)

    # update buffer for query samples
    buffer["0"].extend(ql0)
    buffer["1"].extend(ql1)

    q0 = np.array(training_data)[ql0].tolist()
    q1 = np.array(training_data)[ql1].tolist()

    query = q0 + q1

    return sample, query, buffer






