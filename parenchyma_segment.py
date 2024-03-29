import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK
from skimage import measure, morphology
from pandas import DataFrame
import dicom
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


df = pd.read_excel('./Tuberculosis_classify.xlsx')
# 11
def is_dicom_file(filename):
    # 判断文件是否是dicom格式
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False

def plot_3d(volume, level=-300, spacing=(1.0, 1.0, 1.0),
            face_color=(0.1,0.3,0.8), alpha=0.2, pad = 0,
            fig=None, save_path=None):
    if volume.shape[1:] == (512, 512):
        volume = np.moveaxis(volume, [0,2], [2,0])
    verts, faces, _, _ = measure.marching_cubes_lewiner(volume, level=level, spacing=spacing) # verts, faces, normals, values
    if not fig:
        fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(-pad, volume.shape[0]+pad)
    ax.set_ylim(-pad, volume.shape[1]+pad)
    ax.set_zlim(-pad, volume.shape[2]+pad)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        plt.gcf().canvas.draw()

def load_scan(path):
    print("load_scan")
#     slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    files = os.listdir(path)
    slices = []
    for s in files:
        if is_dicom_file(path + '/' + s):
            instance = pydicom.read_file(path + '/' + s)
            slices.append(instance)
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num+1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key = lambda x:float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    return slices, np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype=np.float32)

def get_pixels_hu(slices):
    print("get_pixels_hu")
    reader = SimpleITK.ImageSeriesReader()
    reader.SetFileNames(SimpleITK.ImageSeriesReader().GetGDCMSeriesFileNames(slices))
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0
    return img_array


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    print("binarize_per_slice")
    bw = np.zeros(image.shape, dtype=bool)
    
    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma, truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma, truncate=2.0) < intensity_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
        
    return bw

def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    print("all_slice_analysis")
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], 
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], 
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume，remove small area
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
#     plt.imshow(label[150],cmap='gray')
#     plt.show()       
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)
            
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def fill_hole(bw):
    print("fill_hole")
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):    
    def extract_main(bw, cover=0.95):
        print("extract_main")
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area)*cover:
                sum = sum+area[count]
                count = count+1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter
           
        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label==properties[0].label

        return bw
    
    def fill_2d_hole(bw):
        print("fill_2d_hole")
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw
    
    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def resampleing(image, scan, new_spacing=[1, 1, 1]):
    # resample the distance between slices
    # spacing = np.array([[scan[0].SliceThickness], scan[0].PixelSpacing], dtype=np.float32)
#     spacing = np.concatenate(([scan[0].SliceThickness], scan[0].PixelSpacing))
    print("resampleing")
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
#     print(spacing)
    resize_factor = spacing / new_spacing
#     print(resize_factor)
    new_real_shape = image.shape * resize_factor
    print(new_real_shape)
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
#     print(real_resize_factor)
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image

#Normaliztion
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    image = (image*255).astype('uint8')
    return image

def step1_python(case_path):
    print("step1_python")
    case ,spacing= load_scan(case_path)
    case_pixels = get_pixels_hu(case_path)
    case_pixels = resampleing(case_pixels,case)
    bw = binarize_per_slice(case_pixels, [1.,1.,1.])
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step
    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing

if __name__ == '__main__':
    
    wb_refine = openpyxl.load_workbook('./Tuberculosis_classify.xlsx')
    sheet_refine = wb_refine.worksheets[0]
    row_num = 0
    for i in range(sheet_refine.max_row+1):
        if i == 0:
            continue
        path = sheet_refine['E'+str(i)].value
        datapath = '/home' + path.split('duqy')[1]
        print(datapath)
        sensity = sheet_refine['D'+str(i)].value
        case_pixels, m1, m2, spacing = step1_python(datapath)
        print(case_pixel.shape())
        volume = np.sum(m1+m2)
        if volume == 0:
            continue
        print(volume)
        img = (m1+m2)*case_pixels
        tmp = img[0]
        h = m1.shape[2]
        w = m1.shape[1]
        tmp = tmp.reshape(1,w,h)
        for k in img:
            if np.sum(k)!=0:
                k = k.reshape(1,w,h)
                tmp = np.vstack([tmp,k])

                if sensity==1:
                    np.save("/data2/duqy/DeepPhthisis/BenMalData/data/tb_210924_overall/sens_{:s}.npy".format(i), tmp[1:,:,:])
                else if sensity==2:
                    np.save("/data2/duqy/DeepPhthisis/BenMalData/data/tb_210924_overall/resis_{:s}.npy".format(i), tmp[1:,:,:])
                else if sensity==3:
                    np.save("/data2/duqy/DeepPhthisis/BenMalData/data/tb_210924_overall/rifam_{:s}.npy".format(i), tmp[1:,:,:])
                else if sensity==4:
                    np.save("/data2/duqy/DeepPhthisis/BenMalData/data/tb_210924_overall/MDR_{:s}.npy".format(i), tmp[1:,:,:])
                else if sensity==5:
                    np.save("/data2/duqy/DeepPhthisis/BenMalData/data/tb_210924_overall/XDR_{:s}.npy".format(i), tmp[1:,:,:])
