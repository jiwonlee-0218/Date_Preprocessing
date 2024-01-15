import os
import torch
import numpy as np
import json
import csv
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from shutil import copyfile

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class CosineLoss(nn.Module):
    def __init__(self, xent=.1, reduction="mean"):
        super(CosineLoss, self).__init__()
        self.xent = xent
        self.reduction = reduction
        self.y = torch.Tensor([1]).cuda()

    def forward(self, input, target):
        cosine_loss = F.cosine_embedding_loss(input, F.one_hot(target, num_classes=input.size(-1)), self.y, reduction=self.reduction).cuda()
        cent_loss = self.cross_entropy(F.normalize(input), target)
        # cent_loss = F.cross_entropy(input, target, reduction=self.reduction).cuda()

        # cent_loss = self.cross_entropy(input, target)
        return cosine_loss + self.xent * cent_loss

# version 1: use torch.autograd
class FocalLossV1(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
        criteria = FocalLossV1()
        logits = torch.randn(8, 19, 384, 384)# nchw, float/half
        lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
        loss = criteria(logits, lbs)
        '''

        # compute loss
        label = F.one_hot(label, num_classes=2)
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss



def l1_regularization(l1_lambda, parameters, device):
    l1_reg = torch.tensor(0).float().to(device)
    for name, params_ in parameters.named_parameters():
        if 'weight' in name:
            l1_reg += l1_lambda * torch.norm(params_.to(device), 1)
    return l1_reg

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_args(savepath, ARGS):
    with open(savepath + 'settings.txt', 'w') as f:
        json.dump(ARGS.__dict__, f, indent=2)
    f.close()

def bestsummary(savepath, fi):

    if fi == 1:
        summary = open(savepath + '/summary.csv', 'w', encoding='utf-8')
        wr_summary = csv.writer(summary)
        wr_summary.writerow(["Epoch", "TRAIN", "", "", "", "VALID", "", "", "", "TEST", "", "", ""])
        wr_summary.writerow(["Epoch", "AUC", "ACC", "SEN", "SPC", "AUC", "ACC", "SEN", "SPC", "AUC", "ACC", "SEN", "SPC",])
    else:
        train_summary = open(savepath + '/summary.csv', 'a')
        wr_summary = csv.writer(train_summary)
    return wr_summary

def bestepoch(auc, acc):

    auc_ep = np.argwhere(auc == np.amax(auc)).flatten().tolist()
    acc_ep = np.argwhere(acc == np.amax(acc)).flatten().tolist()

    if np.size(auc_ep) != 1:
        all_acc = []
        for i in auc_ep:
            all_acc.append(acc[i])
        idx = np.argwhere(all_acc == np.amax(all_acc))
        best_auc_ep = auc_ep[idx[0][0]]
    else:
        best_auc_ep = auc_ep[0]

    if np.size(acc_ep) != 1:
        all_auc = []
        for j in acc_ep:
            all_auc.append(auc[j])
        idx = np.argwhere(all_auc == np.amax(all_auc))
        best_acc_ep = acc_ep[idx[0][0]]
    else:
        best_acc_ep = acc_ep[0]

    return best_auc_ep, best_acc_ep

def bestepoch_acc(acc):
    acc_ep = np.argwhere(acc == np.amax(acc)).flatten().tolist()
    best_acc_ep = acc_ep[0]
    return best_acc_ep

# def remove_file_train(path, bestep, totalepoch):
#     for i in range(totalepoch):
#         if i == bestep:
#             pass
#         else:
#             bestfile = path + 'E%d.pt' % i
#             if os.path.isfile(bestfile):
#                 os.remove(bestfile)
def remove_file_train(path, bestep, epoch):

    for i in range(epoch):
        if i in bestep:
            pass
        elif i == (epoch-1):
            pass
        else:
            bestfile = path + 'E%d.pt' % i
            if os.path.isfile(bestfile):
                os.remove(bestfile)

def cal_all_metric(grou, pred, prob):
    grou = np.hstack(grou)
    acc = accuracy_score(grou, pred)
    auc = roc_auc_score(grou, prob)
    tn, fp, fn, tp = confusion_matrix(grou, pred).ravel()

    sen = tp / (tp + fn)
    spec = tn / (tn + fp)

    # return acc, auc, sen, spec
    return [round(auc, 4), round(acc, 4), round(sen, 4), round(spec, 4)]

def multiclass_metric(grou, pred):

    grou = np.hstack(grou)
    acc = accuracy_score(grou, pred)

    micro_prec = precision_score(grou, pred, average='micro')
    micro_recl = recall_score(grou, pred, average='micro')
    micro_f1sc = f1_score(grou, pred, average='micro')

    macro_prec = precision_score(grou, pred, average='macro')
    macro_recl = recall_score(grou, pred, average='macro')
    macro_f1sc = f1_score(grou, pred, average='macro')

    weight_prec = precision_score(grou, pred, average='weighted')
    weight_recl = recall_score(grou, pred, average='weighted')
    weight_f1sc = f1_score(grou, pred, average='weighted')

    # print('Micro Precision: {:.2f}'.format(precision_score(grou, pred, average='micro')))
    # print('Micro Recall: {:.2f}'.format(recall_score(grou, pred, average='micro')))
    # print('Micro F1-score: {:.2f}\n'.format(f1_score(grou, pred, average='micro')))
    #
    # print('Macro Precision: {:.2f}'.format(precision_score(grou, pred, average='macro')))
    # print('Macro Recall: {:.2f}'.format(recall_score(grou, pred, average='macro')))
    # print('Macro F1-score: {:.2f}\n'.format(f1_score(grou, pred, average='macro')))
    #
    # print('Weighted Precision: {:.2f}'.format(precision_score(grou, pred, average='weighted')))
    # print('Weighted Recall: {:.2f}'.format(recall_score(grou, pred, average='weighted')))
    # print('Weighted F1-score: {:.2f}'.format(f1_score(grou, pred, average='weighted')))
    # print(classification_report(grou, pred, target_names=['Class 1', 'Class 2']))
    # print(confusion_matrix(grou, pred))
    # micro_prec, micro_recl, micro_f1sc, macro_prec, macro_recl, macro_f1sc, weight_prec, weight_recl, weight_f1sc = 0,0,0,0,0,0,0,0,0
    return [acc, micro_prec, micro_recl, micro_f1sc, macro_prec, macro_recl, macro_f1sc, weight_prec, weight_recl, weight_f1sc]


def multiclass_metric_test(grou, pred):

    grou = np.hstack(grou)
    acc = accuracy_score(grou, pred)

    micro_prec = precision_score(grou, pred, average='micro')
    micro_recl = recall_score(grou, pred, average='micro')
    micro_f1sc = f1_score(grou, pred, average='micro')

    macro_prec = precision_score(grou, pred, average='macro')
    macro_recl = recall_score(grou, pred, average='macro')
    macro_f1sc = f1_score(grou, pred, average='macro')

    weight_prec = precision_score(grou, pred, average='weighted')
    weight_recl = recall_score(grou, pred, average='weighted')
    weight_f1sc = f1_score(grou, pred, average='weighted')

    print('Micro Precision: {:.2f}'.format(precision_score(grou, pred, average='micro')))
    print('Micro Recall: {:.2f}'.format(recall_score(grou, pred, average='micro')))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(grou, pred, average='micro')))

    print('Macro Precision: {:.2f}'.format(precision_score(grou, pred, average='macro')))
    print('Macro Recall: {:.2f}'.format(recall_score(grou, pred, average='macro')))
    print('Macro F1-score: {:.2f}\n'.format(f1_score(grou, pred, average='macro')))

    print('Weighted Precision: {:.2f}'.format(precision_score(grou, pred, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(grou, pred, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(grou, pred, average='weighted')))
    print("Classification Reports")
    print(classification_report(grou, pred, target_names=['Class 1', 'Class 2']))
    print("Confusion Matrix")
    print(confusion_matrix(grou, pred))
    # micro_prec, micro_recl, micro_f1sc, macro_prec, macro_recl, macro_f1sc, weight_prec, weight_recl, weight_f1sc = 0,0,0,0,0,0,0,0,0
    return [acc, micro_prec, micro_recl, micro_f1sc, macro_prec, macro_recl, macro_f1sc, weight_prec, weight_recl, weight_f1sc]


def prediction(l_arr):
    # l_arr = torch.from_numpy(np.vstack(l))
    _, pred = torch.max(F.softmax(l_arr, -1).data, -1)
    prob = F.softmax(l_arr, -1).data[:, 1]
    return pred, prob
