#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def acc_metric(pred,labels):
    bsize = pred.shape[0]
    pred_ = pred > 0.5
    acc = np.sum(pred_ == labels) 
    # import pdb;pdb.set_trace()
    acc = acc * 1.0 / bsize
    return acc

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import auc
from opts import parse_opts
from scipy import interp
from sklearn.preprocessing import label_binarize


def confusion_matrics(labels,preds):
    opt = parse_opts()
    # import pdb;pdb.set_trace()
    

    fpr, tpr, thresholds = roc_curve(labels, preds)
    precision, recall, th = precision_recall_curve(labels, preds)
    try:
        tp = tpr[np.where(tpr>0.85)[0][0]]
        spe = 1 - fpr[np.where(tpr>0.85)[0][0]]
        prec = precision[np.where(recall>=0.85)[0][-1]]
    except:
        import pdb;pdb.set_trace()
    auc = roc_auc_score(labels,preds)
    return auc,prec,tp,spe
        
    #print(tpr)
    #print(precision)
    # p = labels.sum()
    # n = labels.shape().sum() - p
    # import pdb;pdb.set_trace()
    
def multiclass_confusion_matrics(labels,preds,probs):
    #mcm = confusion_matrix(labels, preds)
    #print(mcm.shape)
    #fp = mcm.sum(axis=0) - np.diag(mcm)  
    #fn = mcm.sum(axis=1) - np.diag(mcm)
    #tp = np.diag(mcm)
    #tn = mcm.sum() - (fp + fn + tp)

    #recall = tp / (tp + fn)
    #precision = tp / (tp + fp)
    #spec = tn / (tn + fp)
    opt = parse_opts()
    mcm = multilabel_confusion_matrix(labels, preds)
    tn = mcm[:, 0, 0]
    tp = mcm[:, 1, 1]
    fn = mcm[:, 1, 0]
    fp = mcm[:, 0, 1]
    spec = tn / (tn + fp)
    #print(spec)
    precision = precision_score(labels, preds,average="macro")
    recall = recall_score(labels, preds,average="macro")
    spe = np.mean(spec)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    labels_onehot_lst = []
    probs_lst = []
    for label in labels:
        onehot = [0 for i in range(opt.n_classes)]
        onehot[int(label)] = 1
        if len(labels_onehot_lst) == 0:
            labels_onehot_lst = onehot
        else:
            labels_onehot_lst = np.vstack((labels_onehot_lst,onehot))
    
    
    for prob in probs:
        if probs_lst == []:
            probs_lst = prob
        else:
            probs_lst = np.vstack((probs_lst,prob))
    #print(labels_onehot_lst,labels_onehot_lst[:,0],probs_lst,probs_lst[:,0])
    for i in range(opt.n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_onehot_lst[:, i], probs_lst[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        #print(roc_auc[i])
        
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(opt.n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(opt.n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= opt.n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #print(roc_auc["macro"])
        #auc = roc_auc_score(labels,preds,multi_class="ovr")
    return roc_auc["macro"],precision,recall,spe
   
    
    
