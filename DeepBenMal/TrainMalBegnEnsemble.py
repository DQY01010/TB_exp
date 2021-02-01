#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy
import torch
import torchvision
import os
import sys
from torch.nn import DataParallel
from DataIter import MBDataIter
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from model import generate_model
from opts import parse_opts
from layers import *
from metrics import *

from tensorboardX import SummaryWriter
writer = SummaryWriter()

def test_for_ensemble(model, data_loader):
    test_acc = []
    loss_lst = []
    
    pred_lst = []
    label_lst = []
    name_lst = []
    # import pdb;pdb.set_trace()
    for i, (data, target, names) in enumerate(data_loader):
        # import pdb;pdb.set_trace()
        data = Variable(data.cuda(async = True))
        target = Variable(target.cuda(async = True))
        
        out = model(data)
        # cls = loss(out,target)
        # loss_lst.append(cls.data.cpu().numpy())
        pred = torch.sigmoid(out[:,:1])
        pred_arr = pred.data.cpu().numpy()
        label_arr = target.data.cpu().numpy()
        
        pred_lst.append(pred_arr)
        label_lst.append(label_arr)
        name_lst.append(names)
    
    results_dict = {}
    # import pdb;pdb.set_trace()
    for i in range(len(name_lst)):
        for j in range(len(name_lst[i])):
            results_dict[name_lst[i][j]] = (pred_lst[i][j][0],label_lst[i][j][0])
    # import pdb;pdb.set_trace()
    return results_dict

if __name__ == '__main__':
    # Initialize the opts
    opt = parse_opts()
    opt = parse_opts()
    # opt.mean = get_mean(1)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    # opt.sample_duration = 16
    opt.scales = [opt.initial_scale]
    
    multi_results = []
    saved_models_dir = "saved_models/"
    for size_one in os.listdir(saved_models_dir):
        size_lst = size_one.split('_')
        crop_size = int(size_lst[-1])
        if crop_size == 32:
            opt.sample_duration = 8
        tmp_saved_models_dir = saved_models_dir + size_one + "/"
        for res_depth in os.listdir(tmp_saved_models_dir):
            
            depth_lst = res_depth.split('_')
            
            if len(depth_lst) < 2:
                continue
            else:
                depth = int(depth_lst[-1])
                if depth > 30:
                    continue
                
                models_lst  = os.listdir(tmp_saved_models_dir + res_depth )
                models_lst.sort()
                check_path = tmp_saved_models_dir + res_depth + "/" + models_lst[-1]
                checkpoint = torch.load(check_path)
                # import pdb;pdb.set_trace()
                # construct testing data iterator
                opt.sample_size = crop_size
                opt.model_depth = depth
                test_iter = MBDataIter(data_file='test_single.npy',phase='test',crop_size=opt.sample_size,crop_depth=opt.sample_duration)
                test_loader = DataLoader(
                    test_iter,
                    batch_size = opt.batch_size,
                    shuffle = True,
                    num_workers = 32,
                    pin_memory=True)
    
                model, policies = generate_model(opt)
                # import pdb;pdb.set_trace()
                model.load_state_dict(checkpoint['state_dict'])
                model = nn.DataParallel(model.cuda())
                results_dict = test_for_ensemble(model, test_loader)
                
                pred_lst = []
                label_lst = []
                for key in results_dict.keys():
                    results = results_dict[key]
                    pred_lst.append(results[0])
                    label_lst.append(results[1])
                pred_arr = np.array(pred_lst)
                label_arr = np.array(label_lst)
                _acc = acc_metric(pred_arr,label_arr)
                _auc,_prec,_recall = confusion_matrics(label_lst,pred_lst)
                _f1_score = 2 * (_prec * _recall) / (_prec + _recall)
                print("acc %2.4f, auc %2.4f,precision %2.4f,recall %2.4f,f1_score %2.4f!"% (_acc,_auc,_prec,_recall,_f1_score))
                multi_results.append(results_dict)
                print(check_path)
    
    # import pdb;pdb.set_trace()
    # max_ensemble 
    pred_lst = []
    label_lst = []
    for key in multi_results[0].keys():
        cur_pred = []
        cur_label = []
        for i in range(len(multi_results)):
            results = multi_results[i][key]
            cur_pred.append(results[0])
            cur_label.append(results[1])
        pred_lst.append(max(cur_pred))
        label_lst.append(max(cur_label))
    # import pdb;pdb.set_trace()
    max_auc,max_prec,max_recall = confusion_matrics(label_lst,pred_lst)
    max_f1_score = 2 * (max_prec * max_recall) / (max_prec + max_recall)
    pred_arr = np.array(pred_lst)
    label_arr = np.array(label_lst)
    max_acc = acc_metric(pred_arr,label_arr)
    print("acc %2.4f, auc %2.4f,precision %2.4f,recall %2.4f,f1_score %2.4f!"% (max_acc,max_auc,max_prec,max_recall,max_f1_score))
    
    # Mean ensemblec
    
    pred_lst = []
    label_lst = []
    for key in multi_results[0].keys():
        cur_pred = []
        cur_label = []
        for i in range(len(multi_results)):
            results = multi_results[i][key]
            
            cur_pred.append(results[0])
            cur_label.append(results[1])
        pred_lst.append(np.mean(cur_pred))
        label_lst.append(max(cur_label))    
   
    mean_auc,mean_prec,mean_recall = confusion_matrics(label_lst,pred_lst)
    mean_f1_score = 2 * (mean_prec * mean_recall) / (mean_prec + mean_recall)
    pred_arr = np.array(pred_lst)
    label_arr = np.array(label_lst)  
    mean_acc = acc_metric(pred_arr,label_arr)
    print("acc %2.4f, auc %2.4f,precision %2.4f,recall %2.4f,f1_score %2.4f!"% (mean_acc,mean_auc,mean_prec,mean_recall,mean_f1_score))