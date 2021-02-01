#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

import torch
from torch import nn
import math

class Loss(nn.Module):
    def __init__(self, num_hard = 0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # output = output.view(-1, 5)
        # labels = labels.view(-1, 5)
        # import pdb;pdb.set_trace()
        cls = self.classify_loss(outs,labels)
        # import pdb;pdb.set_trace()
        return cls,cls

class AUCPLoss(nn.Module):
    def __init__(self, num_hard = 0, lamb=2,alpha=0.5):
        super(AUCPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.lamb = lamb
        self.alpha = alpha

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # import pdb;pdb.set_trace()
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                print("neg")
            else:
                trans_pos = out_pos.repeat(num_neg,1)
                trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
        except:
            import pdb;pdb.set_trace()
            
        """
            import pdb;pdb.set_trace()
            for i in range(num_pos):
                for j in range(num_neg):
                    penalty_term_sum += (1-(out_pos[i]-out_neg[j])).pow(2)
            import pdb;pdb.set_trace()

            num_pos = np.max((out_pos.shape[0],1))
            num_neg = np.max((out_neg.shape[0],1))
            penalty_term = penalty_term_sum / (2 * num_pos * num_neg)
        """
        
        if outs.data.cpu().numpy().max() is np.nan:
            import pdb;pdb.set_trace()
        
        if labels.data.cpu().numpy().max() is np.nan:
            import pdb;pdb.set_trace()
        
        try:
            
            # print(outs.data.cpu().numpy().max())
            # print(labels.data.cpu().numpy().max())
            cls = self.classify_loss(outs,labels) + self.alpha * penalty_term
        except:
            import pdb;pdb.set_trace()
            
        # import pdb;pdb.set_trace()
        return cls, self.alpha * penalty_term
    
class PAUCPLoss(nn.Module):
    def __init__(self, num_hard = 0, lamb=2,alpha=0.5):
        super(PAUCPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.lamb = 2
        self.alpha = 1

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # import pdb;pdb.set_trace()
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                # print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
                # print("neg")
            else:
                trans_pos = out_pos.repeat(num_neg,1)
                trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                penalty_term = torch.mean((1-(trans_pos-trans_neg)).pow(self.lamb)) / self.lamb
        except:
            import pdb;pdb.set_trace()
            
        """
            import pdb;pdb.set_trace()
            for i in range(num_pos):
                for j in range(num_neg):
                    penalty_term_sum += (1-(out_pos[i]-out_neg[j])).pow(2)
            import pdb;pdb.set_trace()

            num_pos = np.max((out_pos.shape[0],1))
            num_neg = np.max((out_neg.shape[0],1))
            penalty_term = penalty_term_sum / (2 * num_pos * num_neg)
        """
        
        if outs.data.cpu().numpy().max() is np.nan:
            import pdb;pdb.set_trace()
        
        if labels.data.cpu().numpy().max() is np.nan:
            import pdb;pdb.set_trace()
        
        try:
            
            # print(outs.data.cpu().numpy().max())
            # print(labels.data.cpu().numpy().max())
            cls = self.alpha * penalty_term
        except:
            import pdb;pdb.set_trace()
            
        # import pdb;pdb.set_trace()
        return cls, self.alpha * penalty_term
    
class AUCHLoss(nn.Module):
    def __init__(self, num_hard = 0):
        super(AUCHLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # import pdb;pdb.set_trace()
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = torch.mean((1-(trans_pos-trans_neg)))
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = torch.mean((1-(trans_pos-trans_neg)))
                print("neg")
            else:
                trans_pos = out_pos.repeat(num_neg,1)
                trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                penalty_term = torch.mean((1-(trans_pos-trans_neg)))
        except:
            import pdb;pdb.set_trace()
        """
            import pdb;pdb.set_trace()
            for i in range(num_pos):
                for j in range(num_neg):
                    penalty_term_sum += (1-(out_pos[i]-out_neg[j])).pow(2)
            import pdb;pdb.set_trace()

            num_pos = np.max((out_pos.shape[0],1))
            num_neg = np.max((out_neg.shape[0],1))
            penalty_term = penalty_term_sum / (2 * num_pos * num_neg)
        """
        
        cls = self.classify_loss(outs,labels) + 0.1 * penalty_term
        # import pdb;pdb.set_trace()
        return cls, 0.1 * penalty_term

class PAUCLoss(nn.Module):
    def __init__(self, num_hard = 0):
        super(PAUCLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        # import pdb;pdb.set_trace()
        out_pos = outs[labels == 1]
        out_neg = outs[labels == 0]
        penalty_term_sum = 0
        
        try:
            num_pos = out_pos.shape[0]
            num_neg = out_neg.shape[0]
            if num_pos == 0:
                trans_pos = 0
                trans_neg = out_neg
                penalty_term = torch.mean((1-(trans_pos-trans_neg)))
                print("pos")
            elif num_neg == 0:
                trans_pos = out_pos
                trans_neg = 0
                penalty_term = torch.mean((1-(trans_pos-trans_neg)))
                print("neg")
            else:
                trans_pos = out_pos.repeat(num_neg,1)
                trans_neg = out_neg.view([1,num_neg]).t().repeat(1,num_pos)
                penalty_term = torch.mean((1-(trans_pos-trans_neg)))
        except:
            import pdb;pdb.set_trace()
        """
            import pdb;pdb.set_trace()
            for i in range(num_pos):
                for j in range(num_neg):
                    penalty_term_sum += (1-(out_pos[i]-out_neg[j])).pow(2)
            import pdb;pdb.set_trace()

            num_pos = np.max((out_pos.shape[0],1))
            num_neg = np.max((out_neg.shape[0],1))
            penalty_term = penalty_term_sum / (2 * num_pos * num_neg)
        """
        
        cls = penalty_term
        # import pdb;pdb.set_trace()
        return cls, 0.1 * penalty_term
    
class FPLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(FPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        
        h_pos_loss = torch.mul(neg_outs,pos_loss)
        h_neg_loss = torch.mul(outs,neg_loss)
        
        fpcls = - h_pos_loss.mean() - h_neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
            
        return fpcls

class FPLoss1(nn.Module):
    def __init__(self, num_hard=0):
        super(FPLoss1, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        
        h_pos_loss = torch.mul(neg_outs, pos_loss)
        h_neg_loss = torch.mul(outs, neg_loss)
        
        fpcls = - h_pos_loss.mean() - 2 * h_neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
            
        return fpcls
    
class CELoss(nn.Module):
    def __init__(self, num_hard=0):
        super(CELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        fpcls = - pos_loss.mean() - neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
        return fpcls , fpcls

class CWCELoss(nn.Module):
    def __init__(self, num_hard=0):
        super(CWCELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        num_neg = neg_labels.sum()
        num_pos = labels.sum()
        
        Beta_P = num_pos / (num_pos + num_neg)
        Beta_N = num_neg / (num_pos + num_neg)
        # import pdb;pdb.set_trace()
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        fpcls = - Beta_N * pos_loss.mean() - Beta_P * neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
        return fpcls , fpcls    

class RCLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(RCLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        
        h_pos_loss = torch.mul(neg_outs, pos_loss)
        h_neg_loss = torch.mul(outs, neg_loss)
        
        fpcls = - 2 * h_pos_loss.mean() - h_neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
        return fpcls   
    
class FPSimilarityLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(FPLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        
    def forward(self, output, labels, train = True):
        batch_size = labels.size(0)
        outs = self.sigmoid(output[:,:1])
        
        neg_labels = 1 - labels
        neg_outs = 1 - self.sigmoid(output[:,:1])
        
        pos_loss = torch.mul(labels,torch.log(outs))
        neg_loss = torch.mul(neg_labels,torch.log(neg_outs))
        
        h_pos_loss = torch.mul(neg_outs,pos_loss)
        h_neg_loss = torch.mul(outs,neg_loss)
        
        fpcls = - h_pos_loss.mean() - 2 * h_neg_loss.mean()
        
        if fpcls.item() is np.nan:
            import pdb;pdb.set_trace()
            
        return fpcls
    