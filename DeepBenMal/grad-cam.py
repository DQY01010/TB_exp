import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# from skimage import transform
import cv2
import pickle
import os
from glob import glob
from model import generate_model
import sys
import torchvision.transforms as v_transforms
from transform import CenterCrop, Normalize, ToTensor, Resize
import torch
# from grad_cam import GradCAM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import random
sns.set(color_codes=True)
import torch.nn.functional as F
from opts import parse_opts
import argparse


np.set_printoptions(precision=4, suppress=True)

def t_nan(x):
    x[x!=x] = 0.
    return x

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # modules = self.model._modules
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(
            self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = [] # the target model feature maps
        # import pdb; pdb.set_trace()
        for name, module in self.model._modules.items():
            if module ==  self.feature_module:
                # x = module(x)
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)
        
        return target_activations, x


class GradCAM:
    def __init__(self, model,
        feature_module, 
        target_layer_names=['2']
    ):

        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.device = torch.device('cuda')    

        self.extractor = ModelOutputs(
            self.model, 
            self.feature_module, 
            target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, x, index=-1, positive=True):
        features, output = self.extractor(x)
        # print(output)
        prob = torch.sigmoid(output)
#         if not positive:
#             prob = 1. - prob
        print(prob)
        pred = prob[0][index].item()
        print(prob.detach().cpu().data.numpy())
        print(pred)
        L = output.size(-1)
        if index != -1:
            tasks = [index]
        else:
            tasks = list(range(L))
        print(tasks)    
        new_L = len(tasks)
        
        cam = []
        for i in tasks:
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0][i] = 1
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)

            one_hot = one_hot.to(self.device)
            print(one_hot)
            loss = torch.sum(one_hot * output)
            print(loss)
            if not positive:
                loss = -1. * loss
            self.feature_module.zero_grad()
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            grads_val = self.extractor.get_gradients()[-1]
            B, C, D, H, W = grads_val.size()
            target = features[-1] # B * C * D * H * W
            weights = torch.mean(grads_val.view(B, C, -1), dim=-1)
            weights = weights.view(B, C, 1, 1, 1)

            cur_cam = torch.mul(weights, target)
            cur_cam = torch.sum(cur_cam, dim=1, keepdim=True)
            
            cur_cam = F.relu(cur_cam)
            cam.append(cur_cam)
        
#         import pdb; pdb.set_trace()
        cam = torch.cat(cam, dim=1)  # B * L * D * H * W (x.shape[2:])
        # fusion results
        # cam = torch.mean(cam, dim=1, keepdim=True)  # B * 1 * 32 * 48 * 48
        
        cam = F.interpolate(cam, size=x.shape[2:], mode='trilinear', align_corners=True)
        
        reshaped_cam = cam.view(B, new_L, -1)
        min_value = torch.min(reshaped_cam, dim=-1)[0].view(B, new_L, 1, 1, 1)
        cam = cam - min_value

        reshaped_cam = cam.view(B, new_L, -1)
        max_value = torch.max(reshaped_cam, dim=-1)[0].view(B, new_L, 1, 1, 1)
        try:
            assert (max_value==0).sum() == 0, max_value
            cam = cam / max_value
        except:
            max_value[max_value == 0.] = 1e-12
            cam = cam / max_value
        # print(cam.shape)
    
        assert (cam!=cam).sum() == 0, (cam!=cam).sum()
        
        return cam.detach()[0], pred

        
        


os.environ['CUDA_VISIBLE_DEVICES'] = '3'

print('****************************************')

def show_cam_on_image(imgs, masks, save_root):
    alpha = 0.5
    # print(img.shape, mask.shape)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    assert imgs.shape == masks.shape
    for i in range(imgs.shape[0]):
        img = imgs[i]
        mask = masks[i]
        img = img[..., np.newaxis]
        mask = mask
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        # print(img.shape, mask.shape)
        heatmap = np.float32(heatmap) / 255
        cam = alpha * heatmap + (1 - alpha) * np.float32(img)
        cam = cam / np.max(cam)

        save_filename = 'cam_{}.jpg'.format(str(i).zfill(2))
        save_path = os.path.join(save_root, save_filename)
    
        cv2.imwrite(save_path, np.uint8(255 * cam))


class Transform:
    def __init__(self):
        self.trans = v_transforms.Compose([
            Normalize(bound=[-1300., 500.], cover=[0., 1.]),
            CenterCrop([48, 96, 96]),
            ToTensor(), 
            Resize([48, 96, 96]), 
        ])

    def __call__(self, data):
        return self.trans(data)
    
class Config:
    def __init__(self):
        self.sample_size = 224
        self.sample_duration = 48
        self.model_depth = 10
        self.shortcut_type = 'B'
        self.use_dropout = False
        self.activation = 'relu'
        self.att_block = 'none'  
        self.cov = False  
        self.num_classes = 4

        self.ckpt_path = './saved_models/task1/class2_dep10_crop32_aug4_under_FP_0/054.ckpt'
    
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


opt = parse_opts()
device = torch.device('cuda')    
trans = Transform()

#config = Config()

model, policies = generate_model(opt)

ckpt = torch.load('./saved_models/task1/class4/class4_dep10_crop96_dura48_aug4_notover_CEL/0/081.ckpt')
model.load_state_dict(ckpt["state_dict"])
# print(model)
if isinstance(model,torch.nn.DataParallel):
    model = model.module
#     model = model.to(device)

print(model)

model.eval()

grad_cam = GradCAM(
    model, 
    model.layer4, 
    target_layer_names=['0']
)


# prepare data
datapth = '../BenMalData/screenlist/phthisis/nonsolid/0000034622_20180408_HC_0#N1_I05465.npy'
test_data = np.load(datapth)
test_label = [0]

inputs = trans(test_data)
inputs = inputs.to(device)
print('Data shape: ', inputs.shape)


# cam inputs
cam_index = 0

mask, pred = grad_cam(inputs, index=cam_index)
mask_neg, pred = grad_cam(inputs, index=cam_index, positive=False)
print('output shape: ', mask.shape)
print('pred shape: ', pred)


alpha = 0.3

cam_nodule = inputs.squeeze(0).squeeze(0).cpu().numpy()
cam_mask = mask.squeeze(0).squeeze(0).cpu().numpy()
cam_neg_mask = mask_neg.squeeze(0).squeeze(0).cpu().numpy()

# show_cam_on_image(cam_nodule, cam_mask, save_root='./BenMal_CAM/N1_I07181')


for i in range(48):
    cur_nodule = cam_nodule[i]
    cur_mask = cam_mask[i]
    cur_neg_mask = cam_neg_mask[i]
    
    plt.figure(figsize=(9, 3))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.subplot(1, 3, 1)
    plt.imshow(cur_nodule, cmap=plt.cm.gray)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
#     plt.title('{}_nodule'.format(i))

    plt.subplot(1, 3, 2)
    plt.imshow(cur_nodule, cmap=plt.cm.jet, alpha=alpha)
    plt.imshow(cur_mask*255, cmap=plt.cm.jet, alpha=1 - alpha)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(1, 3, 3)
    plt.imshow(cur_nodule, cmap=plt.cm.jet, alpha=alpha)
    plt.imshow(cur_neg_mask*255, cmap=plt.cm.jet, alpha=1 - alpha)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
#     plt.title('{}_cam'.format(i))
    # save_fig_path = os.path.join()
    plt.savefig("./results/task1//class4_dep10_crop96_dura48_aug4_notover_CEL/0000034622_cam_%d.jpg" % (i))   
    plt.show()