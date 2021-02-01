#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os,sys
import time
import cv2
from scipy.ndimage.interpolation import rotate
import scipy.ndimage
import torch.nn.functional as F

class MBDataIterTask1(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData"
        ph_lst = []
        nonph_lst = []
        for i in range(len(self.data_arr)):
            if 'phthisis/' in self.data_arr[i]:
                nonph_lst.append(self.data_arr[i])
            else:
                ph_lst.append(self.data_arr[i])
        
        if phase == "train":
            minus_ben = len(nonph_lst) - len(ph_lst)
            if sample_phase == 'over':
                random.shuffle(ph_lst)
                if minus_ben > len(ph_lst):
                    minus_ben = minus_ben - len(ph_lst)
                    mal_cop = ph_lst[:minus_ben] + ph_lst
                else:
                    mal_cop = ph_lst[:minus_ben]
                self.data_lst = mal_cop * aug + ph_lst * aug + nonph_lst * aug
                
            elif sample_phase == 'under':
                random.shuffle(nonph_lst)
                ben_cop = nonph_lst[:len(ph_lst)]
                self.data_lst = ben_cop + ph_lst
            else:
                random.shuffle(nonph_lst)
                random.shuffle(ph_lst)
                self.data_lst = nonph_lst * aug + ph_lst * aug
        else:
            self.data_lst = nonph_lst + ph_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        #self.resize = Resize(size=[crop_depth,sample_size,sample_size])
        #self.totensor = ToTensor()
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        #label_lst = cur_dir.split('_')
        label = np.zeros((1,),dtype=np.float32)
        
        if 'phthisis' in cur_dir:
            label[0] = 0.0
        else:
            label[0] = 1.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        #print(imgs.shape())
        
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]
        #imgs = self.totensor(imgs)
        #imgs = self.resize(imgs)
        
        return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
        #return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
        
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)

class MBDataIterTask2(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData"
        ph_lst = []
        nonph_lst = []
        for i in range(len(self.data_arr)):
            if 'phthisis/' in self.data_arr[i] or 'infectious/' in self.data_arr[i]:
                nonph_lst.append(self.data_arr[i])
            else:
                ph_lst.append(self.data_arr[i])
        
        if phase == "train":
            minus_ben = len(nonph_lst) - len(ph_lst)
            if sample_phase == 'over':
                random.shuffle(ph_lst)
                if minus_ben > len(ph_lst):
                    minus_ben = minus_ben - len(ph_lst)
                    mal_cop = ph_lst[:minus_ben] + ph_lst
                else:
                    mal_cop = ph_lst[:minus_ben]
                self.data_lst = mal_cop * aug + ph_lst * aug + nonph_lst * aug
                
            elif sample_phase == 'under':
                random.shuffle(nonph_lst)
                ben_cop = nonph_lst[:len(ph_lst)]
                self.data_lst = ben_cop + ph_lst
            else:
                random.shuffle(nonph_lst)
                random.shuffle(ph_lst)
                self.data_lst = nonph_lst * aug + ph_lst * aug
        else:
            self.data_lst = nonph_lst + ph_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        #self.resize = Resize(size=[crop_depth,sample_size,sample_size])
        #self.totensor = ToTensor()
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        #label_lst = cur_dir.split('_')
        label = np.zeros((1,),dtype=np.float32)
        
        if 'phthisis' in cur_dir or 'infectious' in cur_dir:
            label[0] = 0.0
        else:
            label[0] = 1.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        #print(imgs.shape())
        
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]
        #imgs = self.totensor(imgs)
        #imgs = self.resize(imgs)
        
        return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
        #return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
        
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)        
        
class MBDataIterTask3(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData"
        infl_lst = []
        ben_lst = []
        chron_lst = []
        
        for i in range(len(self.data_arr)):
            if 'chronicTissueInflam' in self.data_arr[i]:
                chron_lst.append(self.data_arr[i])
            if 'hamartoma/' in self.data_arr[i] or 'inflammatory_pseudo/' in self.data_arr[i]:
                ben_lst.append(self.data_arr[i])
            else:
                infl_lst.append(self.data_arr[i])
        
        if phase == "train":
            self.data_lst = infl_lst * aug + ben_lst * aug + chron_lst * aug
        else:
            self.data_lst = infl_lst + ben_lst + chron_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((3,),dtype=np.float32)
        
        if 'chronicTissueInflam' in cur_dir:
            label = 2.0
        elif 'hamartoma' in cur_dir or 'inflammatory_pseudo' in cur_dir:
            label = 1.0
        else:
            label = 0.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]
        
        return torch.from_numpy(imgs.astype(np.float32)), label, cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)


class MBDataIterTask4(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData"
        phth_lst = []
        infe_lst = []
        hama_lst = []
        infl_lst = []
        
        for i in range(len(self.data_arr)):
            if 'hamartoma/' in self.data_arr[i]:
                hama_lst.append(self.data_arr[i])
            elif 'inflammatory_pseudo/' in self.data_arr[i]:
                infl_lst.append(self.data_arr[i])
            elif 'infectious/' in self.data_arr[i]:
                infe_lst.append(self.data_arr[i])
            else:
                phth_lst.append(self.data_arr[i])
        
        if phase == "train":
            self.data_lst = phth_lst * aug + hama_lst * aug + infl_lst * aug + infe_lst * aug
        else:
            self.data_lst = phth_lst + hama_lst + infl_lst + infe_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((4,),dtype=np.float32)
        
        if 'inflammatory_pseudo' in cur_dir:
            label = 3.0
        elif 'hamartoma' in cur_dir:
            label = 2.0
        elif 'infectious' in cur_dir:
            label = 1.0
        else:
            label = 0.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]    
        return torch.from_numpy(imgs.astype(np.float32)), label, cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)

class MBDataIterTask5(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData"
        phth_lst = []
        infe_lst = []
        hama_lst = []
        infl_lst = []
        chroc_lst = []
        
        for i in range(len(self.data_arr)):
            if 'hamartoma/' in self.data_arr[i]:
                hama_lst.append(self.data_arr[i])
            elif 'inflammatory_pseudo/' in self.data_arr[i]:
                infl_lst.append(self.data_arr[i])
            elif 'infectious/' in self.data_arr[i]:
                infe_lst.append(self.data_arr[i])
            elif 'chronicTissueInflam/' in self.data_arr[i]:
                chroc_lst.append(self.data_arr[i])
            else:
                phth_lst.append(self.data_arr[i])
        
        if phase == "train":
            self.data_lst = phth_lst * aug + hama_lst * aug + infl_lst * aug + infe_lst * aug + chroc_lst * aug
        else:
            self.data_lst = phth_lst + hama_lst + infl_lst + infe_lst + chroc_lst
        
      
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((5,),dtype=np.float32)
        
        if 'chronicTissueInflam' in cur_dir:
            label = 4.0
        elif 'inflammatory_pseudo' in cur_dir:
            label = 3.0
        elif 'hamartoma' in cur_dir:
            label = 2.0
        elif 'infectious' in cur_dir:
            label = 1.0
        else:
            label = 0.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]    
        return torch.from_numpy(imgs.astype(np.float32)), label, cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)
        
        
class CenterCrop(object):
    def __init__(self, size, zslice):
        assert size in [16,32,48,64,96] and zslice in [6,8,10,16,32,48,64]
        self.size = (int(size), int(size))
        self.zslice = zslice

    def __call__(self,data):
        s, y, x = data.shape
        des_w, des_h = self.size
        des_s = self.zslice
        x_start = max(int(round((x - des_w) / 2.)),0)
        x_end = min(x_start+des_w,x)
        
        y_start = max(int(round((y - des_h) / 2.)),0)
        y_end = min(y_start+des_h, y)
        
        s_start = max(int(round((s - des_s) / 2.)),0)
        s_end = min(s_start+des_s,s)
        
        data = data[s_start : s_end,
                    y_start : y_end,
                    x_start : x_end]
        
        pad_size = (des_s-(s_end-s_start), des_h-(y_end-y_start), des_w-(x_end-x_start))
        pad_edge = ((int(pad_size[0]/2),pad_size[0] - int(pad_size[0]/2)),(int(pad_size[1]/2),pad_size[1] - int(pad_size[1]/2)),(int(pad_size[2]/2),pad_size[2] - int(pad_size[2]/2)))
        
        if np.sum(pad_size) != 0:
            data = np.pad(data, pad_edge, 'edge')
            
        try:
            data = data.reshape(des_s,des_h,des_w)
        except:
            import pdb;pdb.set_trace()
        return data

class RandomCenterCrop(object):
    def __init__(self, size, zslice):
        # import pdb;pdb.set_trace()
        assert size in [16,32,48,64,96] and zslice in [6,8,10,16,32,48,64]
        self.size = (int(size), int(size))
        self.zslice = zslice
        if size == 16:
            self.randseed = 4
        elif size == 32:
            self.randseed = 6
        elif size == 48:
            self.randseed = 8
        elif size == 64:
            self.randseed = 10
        elif size == 96:
            self.randseed = 12
            
    def __call__(self, data):
        s, y, x = data.shape
        des_w, des_h = self.size
        des_s = self.zslice
        
        i = random.randint(-self.randseed, self.randseed)
        j = random.randint(-self.randseed, self.randseed)
        
        x_start = max(int(round((x - des_w) / 2.) + i),0)
        x_end = min(x_start+des_w,x)
        
        y_start = max(int(round((y - des_h) / 2.) + j),0)
        y_end = min(y_start+des_h, y)
        
        s_start = max(int(round((s - des_s) / 2.)),0)
        s_end = min(s_start+des_s,s)
        
        data = data[s_start : s_start + des_s,
                    y_start : y_start + des_h,
                    x_start : x_start + des_w]
        
        pad_size = (des_s-(s_end-s_start), des_h-(y_end-y_start), des_w-(x_end-x_start))
        pad_edge = ((int(pad_size[0]/2),pad_size[0] - int(pad_size[0]/2)),(int(pad_size[1]/2),pad_size[1] - int(pad_size[1]/2)),(int(pad_size[2]/2),pad_size[2] - int(pad_size[2]/2)))
        
        if np.sum(pad_size) != 0:
            data = np.pad(data, pad_edge, 'edge')
            
        data = data.reshape(des_s,des_h,des_w)
        return data
        
class Crop(object):
    def __init__(self,size=48,zslice=16,phase='train'):
        self.crop_size = size
        self.zslice = zslice
        self.phase = phase
        self.random_crop = RandomCenterCrop(size,zslice)
        self.center_crop = CenterCrop(size,zslice)
    
    # normlization of CT values 
    def normlize(self,img):
        MIN_BOUND = -1300
        MAX_BOUND = 500
        img[img>MAX_BOUND] = MAX_BOUND
        img[img<MIN_BOUND] = MIN_BOUND
        # import pdb;pdb.set_trace()
        img = img.astype(np.float32)
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        return img
    
    def __call__(self,img_npy):
        img = np.load(img_npy)
        if self.phase == "test":
            img_r = self.center_crop(img)
        else:
            img_r = self.random_crop(img)
        
        img_r = self.normlize(img_r)
        for shapa_ in img_r.shape[1:]:
            if shapa_ not in [16,32,48,64,96]:
                print(shapa_)
                import pdb;pdb.set_trace()
                
        return img_r
    
class Augmentation(object):
    def __init__(self,phase='train'):
        self.phase = phase

    ## 数据增广    
    def __call__(self,img_r):
        if self.phase == "train":
            ran_type = random.randint(0,1)
            if ran_type:
                angle1 = np.random.rand()*180
                img_r = rotate(img_r,angle1,axes=(1,2),reshape=False,mode='nearest')
                
            ran_type = random.randint(0,1)
            if ran_type:
                img_r = cv2.flip(img_r, 0)
                
            ran_type = random.randint(0,1)
            if ran_type:
                img_r = cv2.flip(img_r, 1)
                
            ran_type = random.randint(0,1)
            if ran_type:
                img_r = np.flip(img_r, 2)
                
        return img_r

class Resize(object):
    def __init__(self, size):
        assert len(size) == 3
        self.z, self.y, self.x = size
    
    def __call__(self, x):
        return F.interpolate(x, size=(self.z, self.y, self.x), mode='trilinear', align_corners=True)

    
class ToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x.copy()).float()
    
    
'''
class MBDataIter(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = ""
        mal_lst = []
        ben_lst = []
        for i in range(len(self.data_arr)):
            if 'Malignant/' in self.data_arr[i]:
                mal_lst.append(self.data_arr[i])
            else:
                ben_lst.append(self.data_arr[i])
        
        if phase == "train":
            minus_ben = len(ben_lst) - len(mal_lst)
            if sample_phase == 'over':
                random.shuffle(mal_lst)
                if minus_ben > len(mal_lst):
                    minus_ben = minus_ben - len(mal_lst)
                    mal_cop = mal_lst[:minus_ben] + mal_lst
                else:
                    mal_cop = mal_lst[:minus_ben]
                self.data_lst = mal_cop * aug + mal_lst * aug + ben_lst * aug
                
            elif sample_phase == 'under':
                random.shuffle(ben_lst)
                ben_cop = ben_lst[:len(mal_lst)]
                self.data_lst = ben_cop + mal_lst
            else:
                random.shuffle(ben_lst)
                random.shuffle(mal_lst)
                self.data_lst = ben_lst * aug + mal_lst * aug
        else:
            self.data_lst = ben_lst + mal_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((1,),dtype=np.float32)
        
        if 'Malignant' in cur_dir:
            label[0] = 1.0
        else:
            label[0] = 0.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]    
        return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)

class MBDataIter2(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = ""
        mal_lst = []
        ben_lst = []
        for i in range(len(self.data_arr)):
            if 'Malignant/' in self.data_arr[i]:
                mal_lst.append(self.data_arr[i])
            else:
                ben_lst.append(self.data_arr[i])
        
        if phase == "train":
            minus_ben = len(ben_lst) - len(mal_lst)
            if sample_phase == 'over':
                random.shuffle(mal_lst)
                if minus_ben > len(mal_lst):
                    minus_ben = minus_ben - len(mal_lst)
                    mal_cop = mal_lst[:minus_ben] + mal_lst
                else:
                    mal_cop = mal_lst[:minus_ben]
                self.data_lst = mal_cop * aug + mal_lst * aug + ben_lst * aug
                
            elif sample_phase == 'under':
                random.shuffle(ben_lst)
                ben_cop = ben_lst[:len(mal_lst)]
                self.data_lst = ben_cop + mal_lst
            else:
                random.shuffle(ben_lst)
                random.shuffle(mal_lst)
                self.data_lst = ben_lst * aug + mal_lst * aug
        else:
            self.data_lst = ben_lst + mal_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((1,),dtype=np.float32)
        
        if 'Malignant' in cur_dir:
            label[0] = 1.0
        else:
            label[0] = 0.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]    
        return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)
'''

"""
class MBDataIter(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData/screenlist"
        ph_lst = []
        nonph_lst = []
        for i in range(len(self.data_arr)):
            if 'nonPhthisis/' in self.data_arr[i]:
                nonph_lst.append(self.data_arr[i])
            else:
                ph_lst.append(self.data_arr[i])
        
        if phase == "train":
            minus_ben = len(nonph_lst) - len(ph_lst)
            if sample_phase == 'over':
                random.shuffle(ph_lst)
                if minus_ben > len(ph_lst):
                    minus_ben = minus_ben - len(ph_lst)
                    mal_cop = ph_lst[:minus_ben] + ph_lst
                else:
                    mal_cop = ph_lst[:minus_ben]
                self.data_lst = mal_cop * aug + ph_lst * aug + nonph_lst * aug
                
            elif sample_phase == 'under':
                random.shuffle(nonph_lst)
                ben_cop = nonph_lst[:len(ph_lst)]
                self.data_lst = ben_cop + ph_lst
            else:
                random.shuffle(nonph_lst)
                random.shuffle(ph_lst)
                self.data_lst = nonph_lst * aug + ph_lst * aug
        else:
            self.data_lst = nonph_lst + ph_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        #self.resize = Resize(size=[crop_depth,sample_size,sample_size])
        #self.totensor = ToTensor()
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        #label_lst = cur_dir.split('_')
        label = np.zeros((1,),dtype=np.float32)
        
        if 'nonPhthisis' in cur_dir:
            label[0] = 0.0
        else:
            label[0] = 1.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        #print(imgs.shape())
        
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]
        #imgs = self.totensor(imgs)
        #imgs = self.resize(imgs)
        
        return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
        #return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
        
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)

class MBDataIterTask1(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData/screenlist"
        phth_lst = []
        hama_lst = []
        inflama_lst = []
        infec_lst = []
        
        for i in range(len(self.data_arr)):
            if 'hamartoma/' in self.data_arr[i]:
                hama_lst.append(self.data_arr[i])
            elif 'inflammatory_pseudo/' in self.data_arr[i]:
                inflama_lst.append(self.data_arr[i])
            elif 'infectious/' in self.data_arr[i]:
                infec_lst.append(self.data_arr[i])
            else:
                phth_lst.append(self.data_arr[i])
        
        if phase == "train":
            self.data_lst = phth_lst * aug + hama_lst * aug + inflama_lst * aug + infec_lst * aug
            '''minus_ben = len(ben_lst) - len(mal_lst)
            if sample_phase == 'over':
                random.shuffle(mal_lst)
                if minus_ben > len(mal_lst):
                    minus_ben = minus_ben - len(mal_lst)
                    mal_cop = mal_lst[:minus_ben] + mal_lst
                else:
                    mal_cop = mal_lst[:minus_ben]
                self.data_lst = mal_cop * aug + mal_lst * aug + ben_lst * aug
                
            elif sample_phase == 'under':
                random.shuffle(ben_lst)
                ben_cop = ben_lst[:len(mal_lst)]
                self.data_lst = ben_cop + mal_lst
            else:
                random.shuffle(ben_lst)
                random.shuffle(mal_lst)
                self.data_lst = ben_lst * aug + mal_lst * aug'''
        else:
            self.data_lst = phth_lst + hama_lst + inflama_lst + infec_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((4,),dtype=np.float32)
        
        if 'infectious' in cur_dir:
            label = 3
        elif 'inflammatory_pseudo' in cur_dir:
            label = 2
        elif 'hamartoma' in cur_dir:
            label = 1
        else:
            label = 0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]
        
        return torch.from_numpy(imgs.astype(np.float32)), label, cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)


class MBDataIterTask2(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData/screenlist"
        phth_lst = []
        hama_lst = []
        inflama_lst = []
        nodule_lst = []
        
        for i in range(len(self.data_arr)):
            if 'hamartoma/' in self.data_arr[i]:
                hama_lst.append(self.data_arr[i])
            elif 'inflammatory_pseudo/' in self.data_arr[i]:
                inflama_lst.append(self.data_arr[i])
            elif 'inflammatoryNodule/' in self.data_arr[i]:
                nodule_lst.append(self.data_arr[i])
            else:
                phth_lst.append(self.data_arr[i])
        
        if phase == "train":
            self.data_lst = phth_lst * aug + hama_lst * aug + inflama_lst * aug + nodule_lst * aug
        else:
            self.data_lst = phth_lst + hama_lst + inflama_lst + nodule_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((4,),dtype=np.float32)
        
        if 'inflammatoryNodule' in cur_dir:
            label = 3
        elif 'inflammatory_pseudo' in cur_dir:
            label = 2
        elif 'hamartoma' in cur_dir:
            label = 1
        else:
            label = 0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]    
        return torch.from_numpy(imgs.astype(np.float32)), label, cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)

class MBDataIterTask3(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData/screenlist"
        phth_lst = []
        hama_lst = []
        inflama_lst = []
        chronic_lst = []
        
        for i in range(len(self.data_arr)):
            if 'hamartoma/' in self.data_arr[i]:
                hama_lst.append(self.data_arr[i])
            elif 'inflammatory_pseudo/' in self.data_arr[i]:
                inflama_lst.append(self.data_arr[i])
            elif 'chronicTissueInflam/' in self.data_arr[i]:
                chronic_lst.append(self.data_arr[i])
            else:
                phth_lst.append(self.data_arr[i])
        
        if phase == "train":
            self.data_lst = phth_lst * aug + hama_lst * aug + inflama_lst * aug + chronic_lst * aug
        else:
            self.data_lst = phth_lst + hama_lst + inflama_lst + chronic_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((4,),dtype=np.float32)
        
        if 'chronicTissueInflam' in cur_dir:
            label = 3.0
        elif 'inflammatory_pseudo' in cur_dir:
            label = 2.0
        elif 'hamartoma' in cur_dir:
            label = 1.0
        else:
            label = 0.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]    
        return torch.from_numpy(imgs.astype(np.float32)), label, cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)        
        
class MBDataIterTask4(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,sample_size=224,aug=1,sample_phase='over'):
        # self.data_dir = data_dir 
        self.phase = phase
        self.data_arr = np.load(data_file)
        self.data_dir = "/home/DeepPhthisis/BenMalData/screenlist"
        phth_lst = []
        hama_lst = []
        inflama_lst = []
        infect_lst = []
        
        for i in range(len(self.data_arr)):
            if 'soild_hamartoma/' in self.data_arr[i]:
                hama_lst.append(self.data_arr[i])
            elif 'soild_inflammatory_pseudo/' in self.data_arr[i]:
                inflama_lst.append(self.data_arr[i])
            elif 'solid_infectious/' in self.data_arr[i]:
                infect_lst.append(self.data_arr[i])
            else:
                phth_lst.append(self.data_arr[i])
        
        if phase == "train":
            self.data_lst = phth_lst * aug + hama_lst * aug + inflama_lst * aug + infect_lst * aug
        else:
            self.data_lst = phth_lst + hama_lst + inflama_lst + infect_lst
            
        random.shuffle(self.data_lst)
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        self.augm = Augmentation(phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))
        
        cur_dir = self.data_dir + self.data_lst[idx]
        label_lst = cur_dir.split('_')
        label = np.zeros((4,),dtype=np.float32)
        
        if 'solid_infectious' in cur_dir:
            label = 3.0
        elif 'solid_inflammatory_pseudo' in cur_dir:
            label = 2.0
        elif 'solid_hamartoma' in cur_dir:
            label = 1.0
        else:
            label = 0.0
            
        if self.phase == "train":
            cur_idx = idx
        else:
            cur_idx = idx # self.test_dict[cur_dir]
        imgs = self.crop(cur_dir)
        
        ## 训练的时候使用数据增广
        if self.phase == "train":
            imgs = self.augm(imgs)
        
        imgs = imgs[np.newaxis,:,:,:]    
        return torch.from_numpy(imgs.astype(np.float32)), label, cur_dir
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)     
"""