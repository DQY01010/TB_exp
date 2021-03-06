import os
import numpy as np
import SimpleITK as sitk
import pydicom
import json
import argparse
import pandas as pd
import xlrd
from tqdm import tqdm
import time
import datetime
import glob
import shutil

reader = sitk.ImageSeriesReader()
fter = sitk.IntensityWindowingImageFilter()
FIELD = ['lung', 'thorax', 'chest', 'recon', 'abdomen', 'ch-abd', 'abd']
studies_dict = {}

def get_slice_thickness(slc1, slc2, slc3):
    thickness = 0.0
    try:
        thickness1 = abs(slc1.SliceLocation - slc2.SliceLocation)
        thickness2 = abs(slc2.SliceLocation - slc3.SliceLocation)
    except:
        return 0.0 
    if abs(thickness1 - thickness2) <= 0.5:
        thickness = thickness1
        return thickness

def get_patient_id(slc):
        patient_id = str(slc.PatientID)
        # patient id need attention cases that begin with 'T'!!!
        if patient_id[0] != 'T':
            patient_id = patient_id.zfill(10)
        return patient_id

def get_date(slc):
    try:
        date = slc.SeriesDate
    except:
        try:
            date = slc.StudyDate
        except:
            try:
                date = slc.AcquisitionDate
            except:
                date = 'missdate'
    return date
    
def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print(path + 'success')
        return True
    else:
        return False

    
    
# CTA_case = []
dicom_dir = "/data/wangg/lung_210301"
dicom_dir_ids = os.listdir(dicom_dir)
dcm_folder = "/data/phthisis/lung_210301"
for dir_ in tqdm(dicom_dir_ids):
    #print(dir_)
    patient_list = glob.glob(os.path.join(dicom_dir,dir_))
    #print(patient_list)
    for case in patient_list:
        ids = sitk.ImageSeriesReader_GetGDCMSeriesIDs(case)
        #print(ids)
        for id_ in ids:
            file_names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(case, id_)
        # file_names = os.listdir(case)
        # series_id = []
            slice_num = len(file_names)
            if slice_num <= 30:
                print('Continue: Current series length less than 30! Series length:', slice_num)
                continue
            print('Continue: Current series length more than 30! Series length:', slice_num)
            dcmf = file_names[0]
            slc1 = pydicom.read_file(dcmf)
            series_description = ''
            body_part_description = ''
            try:
                series_description = slc1.SeriesDescription
            except:
                series_description = ''
            try:
                body_part_description = slc1.BodyPartExamined
            except:
                body_part_description = ''
            try:
                protocol_name = slc1.ProtocolName
            except:
                protocol_name = ''
            incorperate = False
        # print(slc1.SeriesDescription,slc1.BodyPartExamined)
            for f in FIELD:
                if f in series_description.lower() or f in body_part_description.lower() or f in protocol_name.lower():
                    incorperate = True
                    break
            # filter the field
            if not incorperate:
                if series_description == '' and body_part_description == '':
                    # print('Continue: miss field of series description and body part description!')
                    pass
                else:
                    # print('Continue: This is not thorax CT! Description:', series_description, body_part_description)
                    pass
                continue
            try:
                reader.SetFileNames(file_names)
                dcm_image = reader.Execute()
            except:
                # print('Continue: Read current series error!')
                continue
                
            # get other dicom field information
            # patient_id, date, slice_thickness, name
            thickness = 0.0
            try:
                thickness = float(slc1.SliceThickness)
            except:
                slc2 = pydicom.read_file(file_names[1])
                slc3 = pydicom.read_file(file_names[2])
                thickness = get_slice_thickness(slc1,slc2,slc3)

            # slc_thickness.append(thickness)
            horb = ''
            if thickness <= 2.5:
                horb = 'BC'
            elif thickness > 2.5:
                horb = 'HC'
            elif thickness == 0.0:
                # print('Continue: slice thickness cannot acquire!')
                # wrong_lst.append(l+['thickness wrong'])
                continue            
        # patient id
            patient_id = get_patient_id(slc1)
        # date
            date = get_date(slc1)
            patient_name = str(slc1.PatientName)
            # fter.SetWindowMaximum(500)
            # fter.SetWindowMinimum(-1300)
            # itk_img = fter.Execute(dcm_image)
            # img_ary = sitk.GetArrayFromImage(itk_img)
            # print(date,patient_name,patient_id)
            if dcm_image.GetSize()[0] == 512:
                study_name = patient_id + '_' + date + '_' + horb
                print(study_name)
                if study_name not in studies_dict:
                    studies_dict[study_name] = [case, patient_id, date, horb, patient_name]
                else:
                    studies_dict[study_name].append([case, patient_id, date, horb, patient_name])
                for i in range(20):
                    try:
                        os.makedirs(os.path.join(dcm_folder, study_name + '_' + str(i)))
                        outer_num = i
                        break
                    except:
                        continue
                study_name = study_name + '_' + str(outer_num)
                for k in range(len(file_names)):
                    shutil.copyfile(file_names[k], os.path.join(dcm_folder, study_name, str(len(file_names) - k - 1).zfill(3) + '.dcm'))
            else:
                print('Continue: Current series size is not 512*512!')
                continue

# print(studies_dict)
# with open("phth.txt","w") as f :
   # for key,value in studies_dict.items():
       # f.write(key+":"+str(value)+"\n")

# pd.DataFrame.from_dict(studies_dict).to_excel("/home/DeepPhthisis/phth.xls")      

