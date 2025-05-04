import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

"""
NOTES:


metrics of the confusion matrix:
------------------------------------------------------------------------------------------------------------------------

True Positive (TP)  refers to the number of predictions where the classifier correctly predicts the positive class
                    as positive

True Negative (TN)  refers to the number of predictions where the classifier correctly predicts the negative class
                    as negative

False Positive (FP) refers to the number of predictions where the classifier incorrectly predicts the negative class
                    as positive

False Negative (FN) refers to the number of predictions where the classifier incorrectly predicts the positive class
                    as negative


metrics for measuring the models performance
------------------------------------------------------------------------------------------------------------------------

Accuracy (ACC)      overall accuracy meaning the fraction of the total samples that were correctly classified
                    calculated as: (TP+TN) / (TP+TN+FP+FN)

Misclassification Rate (MR)
                    fraction of predictions that were incorrect; also known as classification error
                    calculated as: (1-ACC)

Precision (PRC)     fraction of predictions as a positive class that were actually positive
                    calculated as: TP / (TP+FP)

Recall (REC)        fraction of all positive samples that were correctly predicted as positive; also known as 
                    True Positive Rate (TPR), Sensitivity or Probability of Detection
                    calculated as: TP / (TP+FN)

Specificity (SPC)   fraction of all negative samples that are correctly predicted as negative; also known as
                    True Negative Rate (TNR)
                    calculated as: TN / (TN+FP)

F1 Score            combines precision and recall into a single measure; mathematically it is the harmonic mean of
                    PRC and REC and calculated as: 2 * ((PRC*REC) / (PRC+REC)) == 2*TP / ((2*TP)+FP+FN)


Confusion Matrix in Multi-Classification setting                 
------------------------------------------------------------------------------------------------------------------------
unlike binary classification, there are no positive or negative classes; TP, TN, FP and FN are to be found for each
class individually

e.g. confusion matrix with n=3 classes
                                            [[7, 8, 9]
                                             [1, 2, 3]
                                             [3, 2, 1]]

columns represent the Predicted Classes while rows represent the True Classes; in the diagonal are all TP located; 

individual metrics for Class 1 (C1)
TP = 7 
TN = (2+2+3+1) = 8 
FP = (1+3) = 4          --> 1: falsely predicted Class 2 as C1; 3: misclassified Class 3 as C1
FN = (8+9)= 17          --> 8: falsely predicted C1 as C2; 9: misclassified C1 as C3

doing that for all n=3 classes we get individually PRC, REC and F1, those can be combined to have a single measure for
the whole model.

Micro-F1            calculated by considering the total TP, total FP and total FN of the model; it does not consider each
                    class individually
                    e.g. total FP = (7+2+1) = 10 -- total FP = (1+3) + (8+2) + (9+3) = 26
                         total FN = (8+9) + (1+3) + (3+2) = 26

                         Micro-F1 = 10 / (10+26) = 0.28

Macro-F1            calculates metrics for each class individually and then takes unweighted mean of the measures
                    (as seen above)
                    e.g. F1C1 = 0.4 -- F1C2 = 0.22 -- F1C3 = 0.11

                         Macro-F1 = (0.4+0.22+0.11) / 3 = 0.24

Weighted-F1         unlike Macro-F1 it takes weighted mean of the measures; the weights for each class are the total
                    numbers of samples of that class; since C1=11, C2=12, C3=13

                    e.g. Weighted-F1 = ((0.4*11) + (0.22*12) + (0.11*13)) / (11+12+13) = 0.24
"""


class MetricTracker:

    def __init__(self, criterion, threshold: float = 0.5, mode: str = "weighted"):
        super(MetricTracker, self).__init__()

        self.mode = mode
        self.threshold = threshold
        self.criterion = criterion

    def __metric_scores(self, y_hat, y_true):

        acc = accuracy_score(y_true, y_hat)  # accuracy
        prc = precision_score(y_true, y_hat, average=self.mode)  # precision score
        rec = recall_score(y_true, y_hat, average=self.mode)  # recall score
        f1 = f1_score(y_true, y_hat, average=self.mode)  # f1 score

        return [acc, prc, rec, f1]

    # function to track metrics in multi- and binary-class scenarios
    def compute_metrics(self, y_hat, y_true):

        # cast ground truth and to CPU and get nd.array
        y_true = y_true.cpu().numpy() if not isinstance(y_true, np.ndarray) else y_true
        y_hat = y_hat.detach().cpu() if not isinstance(y_hat, np.ndarray) else y_hat

        # get logits based on selected predictor on class init
        # predictor = nn.Sigmoid()

        # in case of logits
        y_hat_bin = torch.tensor([1 if p >= self.threshold else 0 for p in y_hat]).int()

        return self.__metric_scores(y_hat=y_hat_bin, y_true=y_true)

    # calculating the Hamming score (HS) for multi-label classification
    @staticmethod
    def hamming_score(y_trues: np.array, y_hats: np.array, no_samples: int):

        # set local Hamming score to zero
        local_hamming = []

        # calculate local Hamming score for every data sample
        for i in range(no_samples):
            numerator = float(sum(y_trues[i] & y_hats[i]))
            denominator = float(sum(y_trues[i] | y_hats[i]))

            local_hamming.append(numerator / denominator)

        return local_hamming
