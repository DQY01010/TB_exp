#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

def confusion_matrics(labels,preds):
    
    fpr, tpr, thresholds = roc_curve(labels, preds)
    precision, recall, th = precision_recall_curve(labels, preds)
    auc = roc_auc_score(labels,preds)
    # p = labels.sum()
    # n = labels.shape().sum() - p
    # import pdb;pdb.set_trace()
#     tp = tpr[np.where(thresholds>0.5)[0][-1]]
    tp = tpr[np.where(tpr>0.85)[0][0]]
#     spe = 1 - fpr[np.where(thresholds>0.5)[0][-1]]
    spe = 1 - fpr[np.where(tpr>0.85)[0][0]]
    # tp = recall[np.where(th>=0.5)[0][0]]
    prec = precision[np.where(th>=0.5)[0][0]]
    
    num_n = 0
    num_p = 0
    num_tp = 0
    num_tn = 0
    num_fn = 0
    num_fp = 0
    for i in range(len(labels)):
        if labels[i]:
            num_p+=1
            if preds[i] > 0.5:
                num_tp += 1
            else:
                num_fn += 1
        else:
            num_n+=1
            if preds[i] <0.5:
                num_tn += 1
            else:
                num_fp += 1
                
    tp = num_tp / (num_tp + num_fn)
    spe = num_tn / (num_tn + num_fp)
    prec = num_tp / (num_tp + num_fp)
    # import pdb;pdb.set_trace()
    return auc,prec,tp,spe

results_dir = './results/2019_07_15_3_time/back_10_64_aug4_no_over_CEL/'
results_npy_lst = os.listdir(results_dir)
acc_sum = 0

all_pred = []
all_label = []
for results_npy in results_npy_lst:
    try:
        results = np.load(results_dir+ results_npy).tolist()
        valid_num = int(results_npy.split('_')[1].split('.npy')[0])
        
        max_acc = results['max_acc']
        max_auc = results['max_auc']
        acc_max = results['acc_max']
        auc_max = results['auc_max']
        acc_sum += auc_max
        tmp_pred = results['max_acc_auc_dict']
        for k,v in tmp_pred.items():
            all_label.append(v[1][0])
            all_pred.append(v[0][0])
        print("Valid %d, The max acc is %2.4f, max auc is %2.4f, acc_max is %2.4f, auc_max is %2.4f" % (valid_num, max_acc, max_auc, acc_max, auc_max))
        
    except:
        continue

all_pred_array = np.array(all_pred)
all_label_array = np.array(all_label)

auc,precision,recall,tnr = confusion_matrics(all_label_array,all_pred_array)
acc = (np.sum(all_label_array==(all_pred_array>.5)) / (len(all_pred_array)*1.0))

print(acc)
print(recall)
print(tnr)
print(precision)
print(auc)
print(2 * ((recall * precision)/(recall + precision)))
