import os
import numpy as np
import SimpleITK as sitk
import pydicom
import redis
import json
import argparse
import pandas as pd
import xlrd
from tqdm import tqdm
import time
import datetime
import glob

# FIELD = 
# # FIELD = ['ch', 'abd']

# red = redis.Redis(host='localhost', port=6379, decode_responses=True)
# reader = sitk.ImageSeriesReader()
# fter = sitk.IntensityWindowingImageFilter()

# path = sys.argv[1]

MAXTIME = time.time() - time.mktime(time.strptime('19700101',"%Y%m%d"))

class PatientSeriesProducer():
    def __init__(self,date_filter=None):
        self.FIELD = ['lung', 'thorax', 'chest', 'recon', 'abdomen', 'ch-abd', 'abd']
        self.reader = sitk.ImageSeriesReader()
        self.fter = sitk.IntensityWindowingImageFilter()
        # self.rds = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.rds = redis.Redis(host='localhost', port=6379, db=0)
        self.date_filter = date_filter
        self.date_error = []
        self.null_study = []

    def get_null_study(self):
        return self.null_study


    def __call__(self, data_path, list_name='patient'):
        studies_dict = {}
        try:
            series_ids = self.reader.GetGDCMSeriesIDs(data_path)
        except:
            print('Exception when reading series id!')
        if len(series_ids) == 0:
            print('Original Data Error: no series found in this path. Check if it is empty folder. {}'.format(data_path))
            self.null_study.append(data_path)
            return
        print('Current series length: {}'.format(len(series_ids)))
        for idx, sid in enumerate(series_ids):
            try:
                dcm_names = self.reader.GetGDCMSeriesFileNames(data_path, sid)
            except:
                # read error
                # print('Continue: Read current series file names error!')
                continue
            # the length of the series is not enough for a single study
            slice_num = len(dcm_names)
            if slice_num <= 30:
                # print('Continue: Current series length less than 30! Series length:', slice_num)
                continue
            dcmf = dcm_names[0]
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
            # import pdb; pdb.set_trace()
            for f in self.FIELD:
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
                self.reader.SetFileNames(dcm_names)
                dcm_image = self.reader.Execute()
            except:
                # print('Continue: Read current series error!')
                continue
                
            # get other dicom field information
            # patient_id, date, slice_thickness, name
            thickness = 0.0
            try:
                thickness = float(slc1.SliceThickness)
            except:
                slc2 = pydicom.read_file(dcm_names[1])
                slc3 = pydicom.read_file(dcm_names[2])
                thickness = self.get_slice_thickness(slc1,slc2,slc3)

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
            patient_id = self.get_patient_id(slc1)
            # date
            date = self.get_date(slc1)
            patient_name = str(slc1.PatientName)

            print('Saving the study: series description {}, {}, {}/{}'.format(series_description, horb, idx, len(series_ids)))

            if dcm_image.GetSize()[0] == 512:
                study_name = patient_id + '_' + date + '_' + horb
                if study_name not in studies_dict:
                    studies_dict[study_name] = [[data_path, sid, patient_id, date, horb, patient_name]]
                else:
                    studies_dict[study_name].append([data_path, sid, patient_id, date, horb, patient_name])
            else:
                # print('Continue: Current series size is not 512*512!')
                continue

        # if the patient has no valid study data.
        if not studies_dict:
            print('Original Data Error: no valid study exists in this patient data!')
            return

        # import pdb; pdb.set_trace()
        patient = data_path.split('/')[-1]
        if date_filter:
            try:
                label_cases = self.date_filter(patient, list(studies_dict.keys()))
                print('After filter! The cases that need to be labeled: {}'.format(label_cases))
            except Exception as e:
                self.date_error.append(patient)
                print('Filter data error. To be checked: \n{}'.format(e))
                # rt = self.data_filter.check_patient(patient) 
                # print('patient id {}, find results: {}. studies: \n'.format(patient, rt))
                print('Study keys: \n{}'.format(studies_dict.keys()))
                return
        else:
            print('Date filter is None, all the cases are to be saved!')
            label_cases = list(studies_dict.keys())
        studies = []
        for case in label_cases:
            studies += studies_dict[case]
        print('Finished! Transtorage length: {}.'.format(len(studies)))
        # if len(studies) == 0:
        #     self.null_study.append(patient)

        # save to the redis  
        for study in studies:
            save_data = json.dumps(study)
            # import pdb; pdb.set_trace()
            self.rds.lpush(list_name, save_data)

    @staticmethod
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

    @staticmethod
    def get_patient_id(slc):
        patient_id = str(slc.PatientID)
        # patient id need attention cases that begin with 'T'!!!
        if patient_id[0] != 'T':
            patient_id = patient_id.zfill(10)
        return patient_id

    @staticmethod
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

class DateFilter():
    def __init__(self, excel_file, patient_col, date_col):
        self.pat2date = self.patient_to_operation_date(excel_file, patient_col, date_col)
        print('valid date length {}'.format(len(list(self.pat2date.keys()))))
 
    # def check_patient(patient_id):
    #     if patient_id in self.pat2date:
    #         return time.strftime('%Y-%m-%d %H:%M:%S', self.pat2date[patient_id])
    #     else:
    #         return 'cannot find!'

    # save the patiend_id - operate_date dictionary
    @staticmethod
    def patient_to_operation_date(xlsx_file, patient_col, date_col):
        pat2date = {}
        df = pd.read_excel(xlsx_file, header=None)
        # df = pd.read_excel(xlsx_file)
        patient_key = df.columns[patient_col]
        date_key = df.columns[date_col]
        for idx, row in df.iterrows():
            if str(row[patient_key]) == 'nan' or not isinstance(row[date_key], datetime.datetime):
                continue
            patient_id = str(row[patient_key]).zfill(10)
            date = row[date_key]
            if str(date) == 'nan':
                continue
            # import pdb; pdb.set_trace() 
            date = time.mktime(time.strptime(str(date), "%Y-%m-%d %H:%M:%S"))
            pat2date[patient_id] = date  
        # import pdb; pdb.set_trace()
        return pat2date

    def __call__(self, patient, studies):
        if patient in self.pat2date:
            print('Find patient id {}'.format(patient))
            # cur_dt = time.strftime('%Y-%m-%d %H:%M:%S', self.pat2date[patient])
            # print('Patient id {}, date {}'.format(patient, cur_dt))
        else:
            raise Exception('Can not find this patient pathology label date in xlsx file! {}'.format(patient))
        if len(studies[0].split('_')) != 3:
            raise Exception('This is wrong study format! e.g. {}'.format(studies[0]))
        label_cases = []
        # patients = list(set([p.split('_')[0] for p in studies]))
        if patient not in self.pat2date:
            raise Exception('This patient date error!')
        operation_date = self.pat2date[patient]
        print(time.ctime(operation_date))
        # import pdb; pdb.set_trace()
        hc_date2case = {}
        bc_date2case = {}
        for case in studies:
            date = time.mktime(time.strptime(case.split('_')[1], "%Y%m%d"))
            horb = case.split('_')[-1]
            if horb == 'HC':
                hc_date2case[date] = case
            elif horb == 'BC':
                bc_date2case[date] = case
            else:
                raise Exception('This is wrong study name: {}'.format(case))

        if hc_date2case:
            hc_date_list = list(hc_date2case.keys())
            hc_date_distance = [operation_date - d if operation_date - d >= 0 else MAXTIME for d in hc_date_list]
            if min(hc_date_distance) != MAXTIME:
                label_cases.append(hc_date2case[hc_date_list[hc_date_distance.index(min(hc_date_distance))]])
        # else:
        #     print(hc_date2case, bc_date2case, operation_date)
        if bc_date2case:
            bc_date_list = list(bc_date2case.keys())
            bc_date_distance = [operation_date - d if operation_date - d >= 0 else MAXTIME for d in bc_date_list]
            if min(bc_date_distance) != MAXTIME:
                label_cases.append(bc_date2case[bc_date_list[bc_date_distance.index(min(bc_date_distance))]])
        print('This patient labeled cases: {}'.format(label_cases))
        # else:
        #     print(hc_date2case, bc_date2case, operation_date)
        return label_cases

# # def save_series_study(root):
# class MyThread(Thread):
#     def __init__(self, producer, patient_path_list):
#         super(MyThread, self).__init__()
#         # Thread.__init__(self)
#         self.producer = producer
#         self.patient_path_list = patient_path_list
#         # self.dst = dst
    
#     def run(self):
#         for data_path in self.patient_path_list:
#             self.function(data_path)
    
#     def get_result(self):
#         Thread.join(self)
#         try:
#             return self.result
#         except Exception:
#             return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', default='/data/dicom/Malignant1w', type=str, help='the root folder of the data')

    parser.add_argument('--excel', type=str, help='excel file that include the filter date.')
    parser.add_argument('--patient_col', type=int, help='patient id column in excel.')
    parser.add_argument('--date_col', type=int, help='date column in excel. begin with 0.')
    parser.add_argument('--redis_list', default='patient', type=str, help='')
    
    args = parser.parse_args()
    src = args.s

    excel_file = args.excel
    patient_col = args.patient_col
    date_col = args.date_col
    redis_list = args.redis_list

    # after parse the parameter! the real logic of this bussiness
    date_filter = DateFilter(excel_file, patient_col, date_col)
    if src[0] != '/':
        raise Exception('The parameter value of \'-s\' need to be the absolute path!')

    # initial the producer object
    patient_series_producer = PatientSeriesProducer(date_filter)
    
    # test code here!/data/DeepLN/FromPACS/20190722
    # test_patients = np.load('/data/DeepLN/FromPACS/intermediate_file/test_data_20190926.npy')
    # for patient in tqdm(test_patients[7:]):
    #     data_path = os.path.join(src, patient)
    #     if len(os.listdir(data_path)) == 0:
    #         print('Empty Folder!!!')
    #         continue
    #     patient_series_producer(data_path, redis_list)
    

    print('from numpy file')
    labeled_path = np.load('/root/Downloads/intermediate_files/npy/2020104_task.npy')
    # labeled_path = [os.path.join()]
    # print('from src path {}'.format(src))
    # labeled_path = glob.glob(src + '/*')
    labeled_path = [path for path in labeled_path if len(os.listdir(path)) >= 5]
    # import pdb; pdb.set_trace()
    for data_path in tqdm(labeled_path):
        patient = data_path.split('/')[-1]
        if len(patient) == 10 or patient[0] == 'T':
#             data_path = os.path.join(src, patient)
            if len(os.listdir(data_path)) == 0:
                print('Empty Folder!!!')
            # execute
            patient_series_producer(data_path, redis_list)
        else:
            print('Folder Name Error!!! Folder name {}.'.format(patient))

    null_study_file = '/root/Downloads/intermediate_files/npy/still_null_retain.npy'
    np.save(null_study_file, np.asarray(patient_series_producer.get_null_study()))

     
    date_error_file = '/root/Downloads/intermediate_files/npy/20200104_error.npy'
    np.save(date_error_file, np.asarray(patient_series_producer.date_error))

    # produce !!!
#     for patient in tqdm(os.listdir(src)):
#         if len(patient) == 10 or patient[0] == 'T':
#             data_path = os.path.join(src, patient)
#             if len(os.listdir(data_path)) == 0:
#                 print('Empty Folder!!!')
#             # execute
#             patient_series_producer(data_path, redis_list)
#         else:
#             print('Folder Name Error!!! Folder name {}.'.format(patient))


