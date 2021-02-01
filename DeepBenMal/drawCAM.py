import io
import requests
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import torch
from model import generate_model
from opts import parse_opts
import cv2
import pdb
from torch import nn

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

    
def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name
if __name__ == '__main__':
    # input image
    opt = parse_opts()
    
    #print(opt)
    #LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
    #imgs = np.load("../BenMalData/screenlist/phthisis/nonsolid/0000113986_20161018_BC_0#N1_I06838.npy")
    IMG_URL = 'http://media.mlive.com/news_impact/photo/9933031-large.jpg'
    net, policies = generate_model(opt)

    ckpt = torch.load('./saved_models/task1/class4/class4_dep10_crop96_dura48_aug4_notover_CEL/0/081.ckpt')
    net.load_state_dict(ckpt["state_dict"])
    if isinstance(net,torch.nn.DataParallel):
        net = net.module
    #print(ckpt["state_dict"])
    finalconv_name = "layer4"
    #print(finalconv_name)
    #net = net.to(device)
    net.eval()
    #print(net)
    # hook the feature extractor
    features_blobs = []
    
    net.layer4.register_forward_hook(hook_feature)
    print(net)

    # get the softmax weight
    params = list(net.parameters())
    #print(params)
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
       transforms.Resize((96,96), interpolation=2),
       transforms.ToTensor(),
       normalize
    ])

    """for img in imgs:
        img_tensor = preprocess(img)
        img_variable = Variable(img_tensor.unsqueeze(0))
        print(img_variable)
    logit = net(img_variable)"""

    response = requests.get(IMG_URL)
    img_pil = Image.open(io.BytesIO(response.content))
    img_pil.save('test.jpg')
    print(img_pil)
    img_tensor = preprocess(img_pil)
    print(img_tensor)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit = net(img_variable)
    
    # download the imagenet category list
    #classes = {int(key):value for (key, value)
              #in requests.get(LABELS_URL).json().items()}
    classes = [0]
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread('test.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('CAM.jpg', result)