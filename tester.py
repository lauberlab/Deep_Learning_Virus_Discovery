import torch
import torch.nn as nn
import numpy as np
from utils import (meta_from_loader, move_batch_to_device)
import pandas as pd
from metrics import MetricTracker
from pytorch_metric_learning import distances


class Tester:

    def __init__(self, model, device: str, k: int = None, criterion=None, classify: bool = False,
                 results_dir: str = None):
        super(Tester, self).__init__()

        self.model = model
        self.device = device
        self.criterion = criterion
        self.classify = classify
        self.distance = distances.CosineSimilarity()
        self.metric_tracker = MetricTracker(criterion=criterion)
        self.results_dir = results_dir
        self.k = k

    @torch.no_grad()
    def infer_batch(self, batch_data):

        self.model.eval()

        # output container
        output, labels = [], []

        with torch.no_grad():
            for batch_nr, batch in enumerate(batch_data):

                # compute embeddings
                batch = move_batch_to_device(batch, device=self.device)
                batch_infer = self.model(batch)

                if self.classify:
                    predictor = nn.Sigmoid()
                    batch_infer = predictor(batch_infer)

                output.extend(batch_infer.cpu().numpy())
                labels.extend((batch.y.detach().cpu().numpy()))

            return np.array(output), np.array(labels)

    @torch.no_grad()
    def infer_sample(self, data):

        self.model.eval()

        with torch.no_grad():
            sample = next(iter(data))[0]
            sample = move_batch_to_device(sample, device=self.device)
            inference = self.model(sample)

        return inference.cpu().numpy()

    def compute_similarity_matrix(self, t_emb, q_emb):

        # get distance
        q_dist = self.distance(torch.from_numpy(t_emb), torch.from_numpy(q_emb))

        # rounding
        q_dist = np.around(q_dist.squeeze().numpy(), decimals=8)

        """
        if q_dist.shape[0] > 1:
            ref_idx = np.argmax(q_dist)
            return np.max(q_dist, axis=0), ref_idx

        else:
            return q_dist, 0
        """
        return q_dist

    def test_binary(self, test_loader):

        test_prd = self.infer_batch(test_loader)
        test_labels, test_headers = meta_from_loader(test_loader)
        num_samples = test_prd.shape[0]

        # define output variables
        metrics = self.metric_tracker.compute_metrics(y_true=test_labels, y_hat=test_prd)
        samples = list(range(1, num_samples + 1))
        results = pd.DataFrame(data={"y": [prd for prd in test_prd], "headers": [head for head in test_headers],
                                     "labels": [label for label in test_labels], "samples": samples})

        return results, metrics

    def test_similarity(self, test_loader, template, reference: list):

        # embeddings of the data samples that are to be tested against
        # and the respective header and label information
        test_embeddings, test_labels = self.infer_batch(test_loader)

        # compute the distances between template and test samples
        q_S = self.compute_similarity_matrix(t_emb=template, q_emb=test_embeddings)

        if q_S.shape[0] > 1:
            test_sim = np.max(q_S, axis=0)
            ref_idx = np.argmax(q_S, axis=0)
            ref_O = [reference[idx] for idx in ref_idx]

            ref_O_unq = sorted(list(set(ref_O)))
            ref_O_ctr = [ref_O.count(u) for u in ref_O_unq]
            max_ref_idx = np.argmax(ref_O_ctr)
            max_ref = reference[max_ref_idx]
            max_ref_ctr = ref_O_ctr[max_ref_idx]

        else:
            test_sim = q_S
            max_ref = [reference[0]]
            ref_O = max_ref
            max_ref_ctr = 1

        # determine number of tested samples
        num_samples = test_sim.shape[0]

        # return results in form of dict/dataframe
        samples = list(range(1, num_samples + 1))

        results = pd.DataFrame(data={
            "y": [sim for sim in test_sim],
            # "headers": [head for head in test_headers],
            "labels": [label for label in test_labels],
            "samples": samples,
            "references": ref_O
        })

        return results, max_ref, max_ref_ctr
