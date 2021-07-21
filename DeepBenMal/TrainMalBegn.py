#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy
import torch
# import torchvision
import os
import sys
from torch.nn import DataParallel
from DataIter import MBDataIterTask1,MBDataIterTask2,MBDataIterTask3,MBDataIterTask4,MBDataIterTask5,MBDataIterSensi_Resis
from torch.utils.data import DataLoader
from torch.autograd import Variable
import time
from model import generate_model
from opts import parse_opts
from layers import *
from metrics import *
from sklearn.metrics import confusion_matrix 
from tensorboardX import SummaryWriter
from utils import AverageMeter, calculate_accuracy
from transform import Resize
writer = SummaryWriter()


def train(model, data_loader, optimizer, loss, epoch):
    train_loss = []
    lr = optimizer.param_groups[0]['lr']
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    for i, (data, target, names) in enumerate(data_loader):
        # import pdb;pdb.set_trace()

        # data = Variable(data.cuda(async = True))
        # target = Variable(target.cuda(async = True))

        data = data.cuda(non_blocking=True)
        data = Resize([opt.sample_duration,opt.sample_size,opt.sample_size])(data)
        #print(data.shape)
        #import pdb;pdb.set_trace()
        
        target = target.cuda(non_blocking=True)
        
        # import pdb;pdb.set_trace()
        
        out = model(data)
        #print(out,target)
        #print(out,target)
        if "FP" in opt.save_dir:
            cls = loss(out, target)
        elif "BCE" in opt.save_dir:
            cls = loss(out, target)
        else:
            cls = loss(out,target.long())
        optimizer.zero_grad()
        cls.backward()
        optimizer.step()
        pred = torch.sigmoid(out[:, :1])
        if opt.n_classes == 1:
            train_acc = acc_metric(pred.data.cpu().numpy(),target.data.cpu().numpy())
        else:
            train_acc = calculate_accuracy(out, target.long())
        #train_acc = acc_metric(pred.data.cpu().numpy(), target.data.cpu().numpy())

        try:
            train_loss.append(cls.data[0])
        except:
            train_loss.append(cls.item())

        if i % 5 == 0:
            try:
                print("Training: Epoch %d: %dth batch, loss %2.4f, acc %2.4f, lr: %2.6f!" % (
                epoch, i, cls.item(), train_acc, lr))
            except:
                print("Training: Epoch %d: %dth batch, loss %2.4f, acc %2.4f, lr: %2.6f!" % (
                epoch, i, cls.item(), train_acc, lr))

    return np.mean(train_loss)


def test(model, data_loader, loss, epoch, lr, max_acc, max_auc, acc_max, auc_max, save_recall, save_prec, save_spec, save_f1score):
    test_acc = []
    loss_lst = []

    pred_lst = []
    label_lst = []
    prob_lst = []
    isave = False
    isave_lst = False
    pred_target_dict = {}

    for i, (data, target, names) in enumerate(data_loader):
        # data = Variable(data.cuda(async = True))
        # target = Variable(target.cuda(async = True))
        data = data.cuda(non_blocking=True)
        data = Resize([opt.sample_duration,opt.sample_size,opt.sample_size])(data)
        target = target.cuda(non_blocking=True)

        out = model(data).cuda()
        #print(out,target.long())
        if "FP" in opt.save_dir:
            cls = loss(out, target)
        elif "BCE" in opt.save_dir:
            cls = loss(out, target)
        else:
            cls = loss(out,target.long())
            
        loss_lst.append(cls.data.cpu().numpy())
        if 'FP' in opt.save_dir:
            pred = torch.sigmoid(out[:, :1])
            pred_arr = pred.data.cpu().numpy()
        elif 'BCE' in opt.save_dir:
            pred = torch.sigmoid(out[:, :1])
            pred_arr = pred.data.cpu().numpy()
        else:
            pred = torch.sigmoid(out)
            pred_arr = pred.data.cpu().numpy().argmax(axis=1)
        if opt.n_classes != 1:
            prob_arr = pred.data.cpu().numpy()
        label_arr = target.data.cpu().numpy()
        if opt.n_classes == 1:
            _acc = acc_metric(pred_arr, label_arr)
        else:
            _acc = calculate_accuracy(out, target.long())

        if opt.n_classes == 1:
            for i in range(pred_arr.shape[0]):
                pred_target_dict[names[i]] = [pred_arr[i], label_arr[i]]
        else:
            for i in range(pred_arr.shape[0]):
                pred_target_dict[names[i]] = [pred_arr[i], label_arr[i], prob_arr[i]]
        pred_lst.append(pred_arr)
        label_lst.append(label_arr)
        if opt.n_classes != 1:
            prob_lst.append(prob_arr)
        test_acc.append(_acc)
        # name_lst.append(names)

    # import pdb;pdb.set_trace()
    test_loss = np.mean(loss_lst)
    acc = np.mean(test_acc)
    #print(np.concatenate(label_lst, axis=0))
    if 'FP' in opt.save_dir:
        label_lst = np.concatenate(label_lst, axis=0)[:, 0].tolist()
        pred_lst = np.concatenate(pred_lst, axis=0)[:, 0].tolist()
    else:
        label_lst = np.concatenate(label_lst, axis=0).tolist()
        pred_lst = np.concatenate(pred_lst, axis=0).tolist()
    if opt.n_classes != 1:
        prob_lst = np.concatenate(prob_lst, axis=0).tolist()
    #print(label_lst,pred_lst)
        auc, prec, recall, spec = multiclass_confusion_matrics(label_lst, pred_lst, prob_lst)
    else:
        auc, prec, recall, spec = confusion_matrics(label_lst, pred_lst)
    f1_score = 2 * (prec * recall) / (prec + recall)
    # import pdb;pdb.set_trace()
    if acc > max_acc:
        max_acc = acc
        max_auc = auc
        
        save_recall = recall
        save_prec = prec
        save_spec = spec
        save_f1score = f1_score
        
        isave = True
        isave_lst = True

    if auc > auc_max:
        auc_max = auc
        acc_max = acc
        
        save_recall = recall
        save_prec = prec
        save_spec = spec
        save_f1score = f1_score
        
        isave = True

    print("Testing: Epoch %d:%dth batch, learning rate %2.6f loss %2.4f, acc %2.4f, auc %2.4f,precision %2.4f,recall %2.4f,specificity %2.4f!" % (
        epoch, i, lr, test_loss, acc, auc, prec, recall, spec))
    return max_acc, max_auc, acc_max, auc_max, test_loss, isave, pred_target_dict, isave_lst,save_recall,save_prec,save_spec,save_f1score


if __name__ == '__main__':
    # Initialize the opts
    opt = parse_opts()
    # opt.mean = get_mean(1)
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    # print(opt.arch)
    # opt.sample_duration = 16
    opt.scales = [opt.initial_scale]

    # construct training data iterator
    if opt.task == "task1":
        train_iter = MBDataIterTask1(
            data_file= opt.valid_path + '/train_%d.npy' % opt.num_valid,
            phase='train',
            crop_size=opt.crop_size,
            crop_depth=opt.sample_duration,
            sample_size=opt.sample_size,
            sample_phase=None)
    elif opt.task == "task2":
        train_iter = MBDataIterTask2(
            data_file= opt.valid_path + '/train_%d.npy' % opt.num_valid,
            phase='train',
            crop_size=opt.crop_size,
            crop_depth=opt.sample_duration,
            sample_size=opt.sample_size,
            sample_phase=None)
    elif opt.task == "task3":
        train_iter = MBDataIterTask3(
            data_file= opt.valid_path + '/train_%d.npy' % opt.num_valid,
            phase='train',
            crop_size=opt.crop_size,
            crop_depth=opt.sample_duration,
            sample_size=opt.sample_size,
            sample_phase=None)
    elif opt.task == "task4":
        train_iter = MBDataIterTask4(
            data_file= opt.valid_path + '/train_%d.npy' % opt.num_valid,
            phase='train',
            crop_size=opt.crop_size,
            crop_depth=opt.sample_duration,
            sample_size=opt.sample_size,
            sample_phase=None)
    elif opt.task == "task5":
        train_iter = MBDataIterTask5(
            data_file= opt.valid_path + '/train_%d.npy' % opt.num_valid,
            phase='train',
            crop_size=opt.crop_size,
            crop_depth=opt.sample_duration,
            sample_size=opt.sample_size,
            sample_phase=None)
    elif opt.task == "resis":
        train_iter = MBDataIterSensi_Resis(
            data_file= opt.valid_path + '/train_%d.npy' % opt.num_valid,
            phase='train',
            crop_size=opt.crop_size,
            crop_depth=opt.sample_duration,
            sample_size=opt.sample_size,
            sample_phase=None)
    
    train_loader = DataLoader(
        train_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    # construct testing data iterator
    if opt.task == "task1":
        test_iter = MBDataIterTask1(data_file= opt.valid_path + '/val_%d.npy' % opt.num_valid, phase='test', crop_size=opt.crop_size,
                                    crop_depth=opt.sample_duration, sample_size=opt.sample_size)
    elif opt.task == "task2":
        test_iter = MBDataIterTask2(data_file= opt.valid_path + '/val_%d.npy' % opt.num_valid, phase='test', crop_size=opt.crop_size,
                                    crop_depth=opt.sample_duration, sample_size=opt.sample_size)
    elif opt.task == "task3":
        test_iter = MBDataIterTask3(data_file= opt.valid_path + '/val_%d.npy' % opt.num_valid, phase='test', crop_size=opt.crop_size,
                                    crop_depth=opt.sample_duration, sample_size=opt.sample_size)
    elif opt.task == "task4":
        test_iter = MBDataIterTask4(data_file= opt.valid_path + '/val_%d.npy' % opt.num_valid, phase='test', crop_size=opt.crop_size,
                                    crop_depth=opt.sample_duration, sample_size=opt.sample_size)
    elif opt.task == "task5":
        test_iter = MBDataIterTask5(data_file= opt.valid_path + '/val_%d.npy' % opt.num_valid, phase='test', crop_size=opt.crop_size,
                                    crop_depth=opt.sample_duration, sample_size=opt.sample_size)
    elif opt.task == "resis":
        test_iter = MBDataIterSensi_Resis(data_file= opt.valid_path + '/val_%d.npy' % opt.num_valid, phase='test', crop_size=opt.crop_size,
                                    crop_depth=opt.sample_duration, sample_size=opt.sample_size)
    test_loader = DataLoader(
        test_iter,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True)

    model, policies = generate_model(opt)
    #model = nn.DataParallel(model.cuda(), device_ids=[0])
    
    if opt.n_classes == 1:
        if "FP" in opt.save_dir:
            loss = FPLoss() # Using FPLoss
        else:
            loss = nn.BCEWithLogitsLoss()
    elif "CEL" in opt.save_dir:
        loss = nn.CrossEntropyLoss()

    # loss = Loss()
 
    loss = loss.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt.lr,
        momentum=0.9,
        weight_decay=1e-2)

    max_acc = 0
    max_auc = 0
    acc_max = 0
    auc_max = 0
    save_recall = 0
    save_prec = 0
    save_spec = 0
    save_f1score = 0
    
    max_dict = {}
    max_auc_dict = {}

    #save_dir = "saved_models/" + "size_%d/" % opt.sample_size + opt.model + "_%d" % opt.model_depth + "_%d" %opt.num_valid
    save_dir = "saved_models/" + opt.task + "/class" + str(opt.n_classes) + "/" + opt.save_dir + "/%d" %opt.num_valid
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(opt.start_epoch, opt.epochs):
        train_loss = train(model, train_loader, optimizer, loss, epoch)
        max_acc,max_auc,acc_max,auc_max,test_loss,isave,pred_target_dict,isave_lst,save_recall,save_prec,save_spec,save_f1score = test(model, test_loader, loss, epoch,opt.lr, max_acc,max_auc,acc_max,auc_max,save_recall,save_prec,save_spec,save_f1score)
        if isave_lst:
            max_dict = pred_target_dict

        if isave:
            max_auc_dict = pred_target_dict
            state_dict = model.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'opt': opt},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

        writer.add_scalar('metric_curves/loss/train', train_loss, epoch)
        writer.add_scalar('metric_curves/loss/test', test_loss, epoch)

        print("Epoch %d, the max acc is %2.4f, max auc is %2.4f, the acc max is %2.4f, auc max is %2.4f" % (
        epoch, max_acc, max_auc, acc_max, auc_max))
        print('\n')
        if epoch >= 50 and epoch % 30 == 0:
            opt.lr = opt.lr * 0.5
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=opt.lr,
                momentum=0.9,
                weight_decay=1e-2)
        if opt.task == "task1":
            train_iter = MBDataIterTask1(data_file=opt.valid_path +'/train_%d.npy' % opt.num_valid, phase='train',
                                         crop_size=opt.crop_size, crop_depth=opt.sample_duration, sample_size=opt.sample_size, sample_phase=None)
        elif opt.task == "task2":
            train_iter = MBDataIterTask2(data_file=opt.valid_path +'/train_%d.npy' % opt.num_valid, phase='train',
                                         crop_size=opt.crop_size, crop_depth=opt.sample_duration, sample_size=opt.sample_size, sample_phase=None)
        elif opt.task == "task3":
            train_iter = MBDataIterTask3(data_file=opt.valid_path +'/train_%d.npy' % opt.num_valid, phase='train',
                                         crop_size=opt.crop_size, crop_depth=opt.sample_duration, sample_size=opt.sample_size, sample_phase=None)
        elif opt.task == "task4":
            train_iter = MBDataIterTask4(data_file=opt.valid_path +'/train_%d.npy' % opt.num_valid, phase='train',
                                         crop_size=opt.crop_size, crop_depth=opt.sample_duration, sample_size=opt.sample_size, sample_phase=None)
        elif opt.task == "task5":
            train_iter = MBDataIterTask5(data_file=opt.valid_path +'/train_%d.npy' % opt.num_valid, phase='train',
                                         crop_size=opt.crop_size, crop_depth=opt.sample_duration, sample_size=opt.sample_size, sample_phase=None)
        elif opt.task == "resis":
            train_iter = MBDataIterSensi_Resis(data_file=opt.valid_path +'/train_%d.npy' % opt.num_valid, phase='train',
                                         crop_size=opt.crop_size, crop_depth=opt.sample_duration, sample_size=opt.sample_size, sample_phase=None)
        train_loader = DataLoader(
            train_iter,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True)
    results = {}
    results['max_acc'] = max_acc
    results['max_auc'] = max_auc
    results['acc_max'] = acc_max
    results['auc_max'] = auc_max
    
    results['recall'] = save_recall
    results['prec'] = save_prec
    results['spec'] = save_spec
    results['f1score'] = save_f1score
    
    results['max_dict'] = max_dict
    results['max_auc_dict'] = max_auc_dict
    
    save_results_dir = "results/" + opt.task + "/" + opt.save_dir
    if not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir)
    
    np.save(save_results_dir + "/valid_%d.npy" % opt.num_valid, results)
    print("The max acc is %2.4f, max auc is %2.4f, acc_max is %2.4f, auc_max is %2.4f" % (
    max_acc, max_auc, acc_max, auc_max))
