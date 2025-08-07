import numpy as np
from sklearn import metrics

# Compute Evaluation Metrics with "a range of thresholds on distances".
# Report Accuracy, FRR, FAR, EER, the threshold used to compute EER, and AUC.
def eval_metrics(distances, y):
    # Use sklearn metrics (pos_label=0 is because when distance > threshold --> y_pred=0, which is judged as dissimilar)
    far, tpr, threshold = metrics.roc_curve(y, distances, pos_label=0)
    auc = metrics.auc(far, tpr)
    # false acceptance rate (FAR), the same as false positive rate (FPR)
    # false rejection rate (FRR), the same as false negative rate (FNR)
    frr = 1 - tpr
    # true negative rate (TNR), the same as true rejection rate (TRR)
    tnr = 1 - far

    # The difference of frr and far
    diff = abs(frr - far)
    # Calculate EER when FRR=FAR (the min difference between them) => Find the minimum index
    index = np.argmin(diff)
    eer = (frr[index] + far[index]) / 2
    # The threshold value when FRR=FAR
    eer_threshold = threshold[index]

    # tp: tpr * the number of positive class
    pos_count = np.count_nonzero(y == 0)
    tp = tpr * pos_count
    # tn: tnr * the number of negative class
    neg_count = np.count_nonzero(y == 1)
    tn = tnr * neg_count
    # Accuracy under each threshold
    acc = (tp + tn) / (pos_count + neg_count)
    # Calculate the max accuracy, and get FRR & FAR when max accuracy
    max_idx = np.argmax(acc)
    max_acc = acc[max_idx]
    frr_get = frr[max_idx]
    far_get = far[max_idx]

    return max_acc, frr_get, far_get, eer, eer_threshold, auc