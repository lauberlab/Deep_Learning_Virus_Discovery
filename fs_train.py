import torch
from torch_geometric.data import Dataset, Data, Batch
from sklearn.metrics import confusion_matrix
import numpy as np


class FSTrainer:

    def __init__(self, device, model, optim, loss_fn):

        self.device = device
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn

    def __prepare(self, s_set, q_set):

        print(type(s_set))
        print(type(s_set[0]))

        # batching
        support_batch = Batch.from_data_list(s_set)
        query_batch = Batch.from_data_list(q_set)

        # get the target labels
        target_labels = support_batch.y.to(self.device)

        # model embeddings
        support_features = self.model(support_batch)
        query_features = self.model(query_batch)

        # compute the prototypes for each class in the support set
        prototypes = support_features.mean(dim=0)

        # calculate distances between query features and prototypes
        distances = torch.cdist(query_features, prototypes)

        # predict class labels based on the distances from before
        predicted_labels = torch.argmin(distances, dim=1)

        return target_labels, predicted_labels

    def meta_train(self, support_set: list, query_set: list):

        target_labels, predicted_labels = self.__prepare(support_set, query_set)

        # compute the loss
        loss = self.loss_fn(predicted_labels, target_labels)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss, self.optim

    def meta_test(self, support_set: list, query_set: list):

        target_labels, predicted_labels = self.__prepare(support_set, query_set)

        # distinguish between TP, FP, TN, FN
        tn, fp, fn, tp = confusion_matrix(target_labels, predicted_labels).ravel()

        # precision
        # what proportion of positive identifications was acutally correct?
        # model with no FP has a precision of 1.0
        precision = tp / (tp + fp)

        # recall --> True Positive Rate (TPR)
        # what proportion of actual positives was identified correctly?
        # model with no FN has recall of 1.0
        tpr = tp / (tp + fn)

        # False Positive Rate (FPR)
        fpr = np.array(fp / (fp + tn))

        # AUC for single threshold @ 0.0, considering the points (0,0), (FPR, TPR), and (1,1)
        auc = 0.5 * (tpr * (1 - fpr) + fpr * (1 - tpr) + tpr)

        # compute F1 score
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

        return precision, tpr, fpr, auc, f1
