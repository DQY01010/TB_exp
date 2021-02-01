#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy
import torch
# import torchvision
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

def train(model, data_loader, optimizer, loss, epoch):
    train_loss = []
    lr = optimizer.param_groups[0]['lr']
    for i, (data, target, names) in enumerate(data_loader):
        
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # import pdb;pdb.set_trace()
        out = model(data)
        cls,pen_term = loss(out,target)
        optimizer.zero_grad()
        cls.backward()
        optimizer.step()
        pred = torch.sigmoid(out[:,:1])
        train_acc = acc_metric(pred.data.cpu().numpy(),target.data.cpu().numpy())
        
        try:
            train_loss.append(cls.data[0])
        except:
            train_loss.append(cls.item())
            
        # if epoch == 91:
            # import pdb;pdb.set_trace()
        
        if i % 20 == 0:
            try:
                print("Training: Epoch %d: %dth batch, loss %2.4f, acc %2.4f, lr: %2.6f!" % (epoch,i,cls.data[0],train_acc,lr))
            except:
                print("Training: Epoch %d: %dth batch, loss %2.4f, acc %2.4f, lr: %2.6f!" % (epoch,i,cls.item(),train_acc,lr))
    
    return np.mean(train_loss)    

def test(model, data_loader, loss, epoch, lr, max_acc, max_auc, acc_max, auc_max):
    test_acc = []
    loss_lst = []
    
    pred_lst = []
    label_lst = []
    isave = False
    isave_lst = False
    pred_target_dict = {}
    
    for i, (data, target, names) in enumerate(data_loader):
        # data = Variable(data.cuda(async = True))
        # target = Variable(target.cuda(async = True))
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        out = model(data)
        cls,pen_term = loss(out,target)
        loss_lst.append(cls.data.cpu().numpy())
        pred = torch.sigmoid(out[:,:1])
        pred_arr = pred.data.cpu().numpy()
        label_arr = target.data.cpu().numpy()
        _acc = acc_metric(pred_arr,label_arr)
        
        for i in range(pred_arr.shape[0]):
            pred_target_dict[names[i]] = [pred_arr[i], label_arr[i]]
        pred_lst.append(pred_arr)
        label_lst.append(label_arr)
        test_acc.append(_acc)
        # print(pred_arr.shape)
        # name_lst.append(names)
        
    
    # import pdb;pdb.set_trace()
    test_loss = np.mean(loss_lst)
    
    label_lst = np.concatenate(label_lst,axis=0)[:,0].tolist()
    pred_lst = np.concatenate(pred_lst,axis=0)[:,0].tolist()
    auc,prec,recall,spec = confusion_matrics(label_lst,pred_lst)
    f1_score = 2 * (prec * recall) / (prec + recall)
    
    label_arr0 = np.array(label_lst)
    pred_arr0 = np.array(pred_lst)
    acc = acc_metric(pred_arr0, label_arr0)
    
    # import pdb;pdb.set_trace()
    if acc > max_acc:
        max_acc = acc
        max_auc= auc
        isave = True
        isave_lst = True
        
    elif acc == max_acc and auc > max_auc:
        max_acc = acc
        max_auc= auc
        isave = True
        isave_lst = True
        
    if auc > auc_max:
        auc_max = auc
        acc_max = acc
        isave = True
        
    print("Testing: Epoch %d:%dth batch, learning rate %2.6f loss %2.4f, acc %2.4f, auc %2.4f,precision %2.4f,recall %2.4f!" % (epoch,i,lr,test_loss,acc,auc,prec,recall))
    return max_acc,max_auc,acc_max,auc_max,test_loss,isave,pred_target_dict,isave_lst
    
if __name__ == '__main__':
    # Initialize the opts
    opt = parse_opts()
    # opt.mean = get_mean(1)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.scales = [opt.initial_scale]
    # import pdb;pdb.set_trace()
    train_iter = MBDataIter(
        data_file='./valid_5/train_%d.npy'%opt.num_valid,
        phase='train',
        crop_size=opt.sample_size,
        crop_depth=opt.sample_duration,
        aug=opt.aug,
        sample_phase=opt.sample)
    
    train_loader = DataLoader(
        train_iter,
        batch_size = opt.batch_size,
        shuffle = True,
        num_workers = 32,
        pin_memory=True)
    
    # construct testing data iterator
    test_iter = MBDataIter(
        data_file='./valid_5/val_%d.npy'%opt.num_valid,
        phase='test',
        crop_size=opt.sample_size,
        crop_depth=opt.sample_duration)
    
    test_loader = DataLoader(
        test_iter,
        batch_size = opt.batch_size,
        shuffle = True,
        num_workers = 32,
        pin_memory=True)
    
    model, policies = generate_model(opt)
    model = nn.DataParallel(model.cuda())
    
    # import pdb;pdb.set_trace()
    if "FP" in opt.save_dir:
        if "FP1" in opt.save_dir:
            loss = FPLoss1()
        else:
            loss = FPLoss()
    elif "RC" in opt.save_dir:
        loss = RCLoss()
    elif "AUCP" in opt.save_dir:
        loss = AUCPLoss()
    elif "AUCH" in opt.save_dir:
        print("AUCH")
        loss = AUCHLoss()
    else:
        loss = Loss()
    loss = loss.cuda()
    
    optimizer = torch.optim.SGD(
         model.parameters(),
         lr=opt.lr,
         momentum = 0.9,
         weight_decay = 1e-4)
    
    max_acc = 0
    max_auc = 0
    acc_max = 0
    auc_max = 0
    
    max_dict = {}
    max_auc_dict = {}
    target_dir = opt.save_dir
    save_dir = "saved_models/" + target_dir + "/size_%d/" % opt.sample_size + opt.model + "_%d" % opt.model_depth
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for epoch in range(opt.start_epoch,opt.epochs):
        train_loss = train(model, train_loader, optimizer, loss, epoch)
        max_acc,max_auc,acc_max,auc_max,test_loss,isave,pred_target_dict,isave_lst = test(model, test_loader, loss, epoch,opt.lr, max_acc,max_auc,acc_max,auc_max)
        if isave_lst:
            max_dict = pred_target_dict
            
        if isave:
            # max_auc_dict = pred_target_dict
            state_dict = model.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'opt': opt},
                os.path.join(save_dir, '%03d.ckpt' % epoch))
                
        writer.add_scalar('metric_curves/loss/train',train_loss,epoch)
        writer.add_scalar('metric_curves/loss/test',test_loss,epoch)
        
        print ("Epoch %d, the max acc is %2.4f, max auc is %2.4f, the acc max is %2.4f, auc max is %2.4f" %(epoch, max_acc, max_auc,acc_max,auc_max))
        print ('\n')
        if epoch >= 50 and epoch % 30 == 0:
            opt.lr = opt.lr * 0.1
            optimizer = torch.optim.SGD(
                     model.parameters(),
                     lr=opt.lr,
                     momentum = 0.9,
                     weight_decay = 1e-4)
            
        train_iter = MBDataIter(
            data_file='./valid_5/train_%d.npy'%opt.num_valid,
            phase='train',
            crop_size=opt.sample_size,
            crop_depth=opt.sample_duration,
            aug=opt.aug,
            sample_phase=opt.sample)
        
        train_loader = DataLoader(
            train_iter,
            batch_size = opt.batch_size,
            shuffle = True,
            num_workers = 32,
            pin_memory=True)
        
    results = {}
    results['max_acc'] = max_acc
    results['max_auc'] = max_auc
    results['acc_max'] = acc_max
    results['auc_max'] = auc_max
    
    results['max_dict'] = max_dict
    results['max_auc_dict'] = max_auc_dict
    results_dict_path = "results/"+target_dir
    if not os.path.exists(results_dict_path):
        os.makedirs(results_dict_path)
    np.save("results/"+target_dir+"/valid_%d.npy" % opt.num_valid, results)
    print("The max acc is %2.4f, max auc is %2.4f, acc_max is %2.4f, auc_max is %2.4f" %(max_acc, max_auc, acc_max, auc_max))