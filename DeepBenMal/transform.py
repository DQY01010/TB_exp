import torch
import numpy as np
import random

import torch.nn.functional as F


class ToTensor(object):

    def __call__(self, x):
        return torch.from_numpy(x.copy()).float()

class Resize(object):
    def __init__(self, size):
        assert len(size) == 3
        self.z, self.y, self.x = size
    
    def __call__(self, x):
        return F.interpolate(x, size=(self.z, self.y, self.x), mode='trilinear', align_corners=True)
    
class CenterCrop(object):
    def __init__(self, size):
        if not isinstance(size, list):
            raise Exception("Size must be a list object")
        if len(size) != 3:
            raise Exception("Size length must be 3. (z, y, x)")
        self.size = size

    def __call__(self, data):
        z, y, x = data.shape
        des_z, des_y, des_x = self.size
        start_z = int(round((z - des_z) / 2.))
        start_y = int(round((y - des_y) / 2.))
        start_x = int(round((x - des_x) / 2.))
        data = data[start_z : start_z + des_z, 
                   start_y : start_y + des_y, 
                   start_x : start_x + des_x]
        data = data[np.newaxis, :]
        data = data[np.newaxis, :]
        return data



class Normalize(object):
    def __init__(self, bound=[-1300.0, 500.0], cover=[0.0, 1.0]):
        if not isinstance(bound, list):
            raise Exception("In transform/Normalize.. bound must be a list.")
        self.minbound = min(bound)
        self.maxbound = max(bound)
        if not isinstance(cover, list):
            raise Exception("In transform/Normalize.. cover must be a list.")
        self.target_min = min(cover)
        self.target_max = max(cover)

    def __call__(self, x):
        out = (x - self.minbound) / (self.maxbound - self.minbound)
        out[out>self.target_max] = self.target_max
        out[out<self.target_min] = self.target_min
        return out

class TripleCenterCrop(object):
    def __init__(self, sizes):
        self.sizes = {}
        self.sizes['small'] = sizes[0]
        self.sizes['middle'] = sizes[1]
        self.sizes['large'] = sizes[2]
        self.totensor = ToTensor()

    def __call__(self, data):
        nodule = {}
        for key, value in self.sizes.items():
            sample_size = [value[1], value[0], value[0]]
            crop = CenterCrop(sample_size)
            temp = crop(data)
            nodule[key] = self.totensor(temp)
        return nodule

class TripleRandomCrop(object):
    def __init__(self, sizes):
        self.sizes = {}
        self.sizes['small'] = sizes[0]
        self.sizes['middle'] = sizes[1]
        self.sizes['large'] = sizes[2]
        self.totensor = ToTensor()

    def __call__(self, data):
        nodule = {}
        # print(data)
        for key, value in self.sizes.items():
            sample_size = [value[1], value[0], value[0]]
            crop = RandomCrop(sample_size)
            temp = crop(data)
            nodule[key] = self.totensor(temp)
        return nodule

class RandomCrop(object):
    def __init__(self, size):
        if not isinstance(size, list):
            raise Exception('Error in Random crop!')
        self.size = size
        self.randseed = np.floor(np.asarray(size) // 8)

    def __call__(self, data):
        s, y, x = data.shape
       
        des_s, des_y, des_x = self.size
        i = random.randint(-self.randseed[2], self.randseed[2])
        j = random.randint(-self.randseed[1], self.randseed[1])
        k = random.randint(-self.randseed[0], self.randseed[0])

        x_start = int(round((x - des_x)/2.) + i)
        y_start = int(round((y - des_y)/2.) + j)
        s_start = int(round((s - des_s)/2.) + k)
        data = data[s_start : s_start + des_s, 
                    y_start : y_start + des_y, 
                    x_start : x_start + des_x]
        data = data[np.newaxis, :]
        return data

class RandomFlip(object):
    def __call__(self, data):
        if len(data.shape) == 3:
            base = 0
        elif len(data.shape) == 4:
            base = 1
        else:
            raise Exception('Random Flip Error!')
        if random.random() < 0.5: 
            data = np.flip(data, base)
        if random.random() < 0.5:
            data = np.flip(data, base+1)
        if random.random() < 0.5:
            data = np.flip(data, base+2)

        return data

class RandomRotation(object):
    def __call__(self, data):
        assert len(data.shape) == 3, 'data shape: ' + str(data.shape)
        axial_rot_num = random.randint(0, 3)
        sag_rot_num = random.randint(0, 1)
        cor_rot_num = random.randint(0, 1)
        data = np.rot90(data, k=axial_rot_num, axes=(1, 2))
        data = np.rot90(data, k=sag_rot_num*2, axes=(0, 1))
        data = np.rot90(data, k=sag_rot_num*2, axes=(0, 2))
        return data







