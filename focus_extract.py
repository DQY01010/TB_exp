import openpyxl
import numpy as np
import os
import pydicom
import SimpleITK
import matplotlib.pyplot as plt
import scipy.ndimage

def is_dicom_file(filename):
    # 判断文件是否是dicom格式
    file_stream = open(filename, 'rb')
    file_stream.seek(128)
    data = file_stream.read(4)
    file_stream.close()
    if data == b'DICM':
        return True
    return False

def load_scan(path):
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
    reader = SimpleITK.ImageSeriesReader()
    reader.SetFileNames(SimpleITK.ImageSeriesReader().GetGDCMSeriesFileNames(slices))
    image = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(image)
    img_array[img_array == -2000] = 0
    return img_array

def resampleing(image, spacing, new_spacing=[1, 1, 1]):
#     spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    print(new_real_shape)
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image

if __name__ == '__main__':
    wb_refine = openpyxl.load_workbook('./Tuberculosis_anno.xlsx')
    sheet_refine = wb_refine.worksheets[0]
    row_num = 0
    for i in range(sheet_refine.max_row+1):
#     for i in range(2): #test
        if row_num == 0:
            row_num += 1
        else:
            datapath = sheet_refine['H'+str(i)].value
#             datapath = './0000032147_20100612/'
            overall_id = sheet_refine['D'+str(i)].value
            diamter = sheet_refine['G'+str(i)].value
            radius = math.ceil(diamter/2)
            location = sheet_refine['E'+str(i)].value
            slice_idx = sheet_refine['F'+str(i)].value
            x1 = int(location.split(',')[0].split('(')[1])
            y1 = int(location.split(',')[1])
            x2 = int(location.split(',')[2])
            y2 = int(location.split(',')[3].split(')')[0])
            slices, spacing = load_scan(datapath)
            img_array = get_pixels_hu(datapath)
            print(img_array.shape,spacing)
            if slice_idx - radius < 0:
                axis_z_min = 0
            else:
                axis_z_min = slice_idx - radius
            if slice_idx + radius > img_array.shape[0]:
                axis_z_max = img_array.shape[0]
            else:
                axis_z_max = slice_idx + radius
            nodule_anno = img_array[axis_z_min:axis_z_max,x1:x2,y1:y2]
#             resample_nodule = resampleing(nodule_anno,spacing)
#             print(nodule_anno.shape,resample_nodule.shape)
#             plt.imshow(resample_nodule[5],cmap='gray')
#             plt.show()
#             plt.imshow(nodule_anno[5],cmap='gray')
#             plt.show()
            np.save('/data2/duqy/DeepPhthisis/BenMalData/data/tb_210301_refine/'
                    +datapath.split('/')[-1]+'#'+overall_id.split('I')[0]+'_I'+overall_id.split('I')[1]+'.npy' ,nodule_anno)
            row_num += 1