import os
import numpy as np
import scipy
from scipy.ndimage.interpolation import rotate

if __name__ == "__main__":
    path = '/home/DeepPhthisis/BenMalData/data/tb_210301_refine/'
    dir_list = os.listdir(path)
    for i in dir_list:
        image_array = np.load(path+i)
        print(image_array.shape)
        real_resize_factor = [32/image_array.shape[0],64/image_array.shape[1],64/image_array.shape[2]]
        image = scipy.ndimage.interpolation.zoom(image_array, real_resize_factor, mode='nearest')
        label = np.load('/home/DeepPhthisis/BenMalData/lung_210301_label/' + i)
        if label[0] == '1':
            np.save("/home/DeepPhthisis/BenMalData/data/TB_210301_resize/sensitivity/"+i,image)
        elif label[0] == '':
            continue
        else:
            np.save("/home/DeepPhthisis/BenMalData/data/TB_210301_resize/resistant/"+i,image)
