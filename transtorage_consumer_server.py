import  redis
import os
from threading import Thread
import argparse
import json
import SimpleITK as sitk
import imageio
import shutil

# from tqdm import tqdm

# class TranstorageConsumer(Thread):
#     def __init__(self):
#         pass

#     def run(self):
#         pass
reader = sitk.ImageSeriesReader()
fter = sitk.IntensityWindowingImageFilter()

def exe(study, dst):
    # wrong_lst = []
    dcm_folder = os.path.join(dst,'dcms')
    jpg_folder = os.path.join(dst,'jpgs')
    dcm_names = reader.GetGDCMSeriesFileNames(study[0], study[1])
    study_name = study[2] + '_' + study[3] + '_' + study[4]            
    for i in range(20):
        try:
            os.makedirs(os.path.join(dcm_folder, study_name + '_' + str(i)))
            os.makedirs(os.path.join(jpg_folder, study_name + '_' + str(i)))
            outer_num = i
            break
        except:
            continue
    study_name = study_name + '_' + str(outer_num)

    # thepath = os.path.join(dst, 'dcms', study_name)
    print('Curent study: {}'.format(study_name))
    try:
        reader.SetFileNames(dcm_names)
        dcm_image = reader.Execute()
        fter.SetWindowMaximum(500)
        fter.SetWindowMinimum(-1300)
        itk_img = fter.Execute(dcm_image)
        img_ary = sitk.GetArrayFromImage(itk_img)
        for j in range(len(img_ary)):
            imageio.imwrite(os.path.join(jpg_folder, study_name, str(j).zfill(3) + '.jpg'), img_ary[len(img_ary) - j - 1], format='jpeg')
        for k in range(len(dcm_names)):
            shutil.copyfile(dcm_names[k], os.path.join(dcm_folder, study_name, str(len(dcm_names) - k - 1).zfill(3) + '.dcm'))
        
        mediastinum_path = os.path.join(jpg_folder, study_name, 'med')
        if not os.path.exists(mediastinum_path):
            os.makedirs(mediastinum_path)
        fter.SetWindowMaximum(225)
        fter.SetWindowMinimum(-125)
        itk_img = fter.Execute(dcm_image)
        img_ary = sitk.GetArrayFromImage(itk_img)
        for j in range(len(img_ary)):
            imageio.imwrite(os.path.join(mediastinum_path, str(j).zfill(3) + '.jpg'), img_ary[len(img_ary) - j - 1], format='jpeg')
    except:
        print(study)
        # continue
        # import pdb; pdb.set_trace()
        # wrong_lst.append(l+['read date error'])
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=str, help='the destination folder of the data')
    parser.add_argument('--redis_list', default='patient', type=str, help='')
    args = parser.parse_args()
    dst = args.t
    redis_list = args.redis_list

    if dst[0] != '/':
        raise Exception('The parameter value of \'-t\' need to be the absolute path!')

    # pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True, db=0)
    pool = redis.ConnectionPool(host='localhost', port=6379, db=0)
    r = redis.Redis(connection_pool=pool)
    print('-----------------START!!!!---------------------')
    print('The data tranformation service is start!')
    while True:
        # import pdb;pdb.set_trace()
        task = r.brpop(redis_list, 0)
        study = json.loads(task[1])
        # import pdb; pdb.set_trace()
        exe(study, dst)
