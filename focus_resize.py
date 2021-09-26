import os
import numpy as np

if __name__ = "__main__":
    path = '/home/DeepPhthisis/BenMalData/tb_210301_refine/'
    dir_list = os.dir_list(path)
    for i in dir_list:
        image_array = np.load(path+i)
        print(image_array.shape)
        new_shape = [64,128,128]
        real_resize_factor = new_shape / image_array.shape
        new_spacing = spacing / real_resize_factor
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        np.save("/home/DeepPhthisis/BenMalData/data/TB_210301_resize/"+i,image)