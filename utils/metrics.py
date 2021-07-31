import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score


class DiceLoss(nn.Module):

    def init(self):
        super(DiceLoss, self).init()

    def forward(self, pred, target):
        smooth = 1.

        return 1 - self.dice_coef(pred, target, smooth=smooth)

    def dice_coef(self, pred, target, smooth=1e-3):
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return (2. * intersection + smooth) / (A_sum + B_sum + smooth)

    def jaccard_coef(self, pred, target, smooth=1e-3):
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return (intersection + smooth) / (A_sum + B_sum - intersection + smooth)


def bestThresshold(y_true, y_pred):
    best_thresh = None
    best_score = 0
    for thresh in np.arange(0.1, 0.901, 0.01):
        score = fbeta_score(y_true, np.array(y_pred) > thresh, beta=1)
        if score > best_score:
            best_thresh = thresh
            best_score = score

    return best_score, best_thresh


def cal_metrics(y_pred, y_true, thresh_val=0.5):

    true_positive = int(((y_true == 1.0)*(y_pred > thresh_val)*1.0).sum())
    false_negative = int(((y_true == 1.0)*(y_pred < thresh_val)*1.0).sum())
    false_positive = int(((y_true == 0.0)*(y_pred > thresh_val)*1.0).sum())
    true_negative = int(((y_true == 0.0)*(y_pred < thresh_val)*1.0).sum())

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    recall = sensitivity
    precision = true_positive / (true_positive + false_positive)
    accuracy = (true_positive + true_negative) / y_pred.shape[0]

    return sensitivity, specificity, recall, precision, accuracy, (true_positive, false_negative, true_negative, false_positive)


def cal_auc(y_g, y_p):

    return roc_auc_score(y_g, y_p)


def report_result(y_g, y_p):
    best_f1, best_thresh = bestThresshold(y_g, y_p)
    print('Highest F1 score:', best_f1)
    print('Optimal threshold value (base on F1 score):', best_thresh)
    print('--------------------------------------------------------------------')
    best_thresh = 0.5

    sensitivity, specificity, recall, precision, accuracy, (
        true_positive, false_negative, true_negative, false_positive) = cal_metrics(y_p, y_g, thresh_val=best_thresh)
    print('True positive:', true_positive)
    print('False negative:', false_negative)
    print('True negative:', true_negative)
    print('False positive:', false_positive)
    print('--------------------------------------------------------------------')
    print('Sensitivity/Recall:', sensitivity)
    print('Specificity:', specificity)
    print('Precision:', precision)
    print('Accuracy:', accuracy)
    print('--------------------------------------------------------------------')
    auc = roc_auc_score(y_g, y_p)
    f1 = fbeta_score(y_g, 1.0*(y_p > best_thresh), beta=1)
    print('ROC AUC score:', auc)
    print('F1 score:', f1)
