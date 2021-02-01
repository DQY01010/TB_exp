#！/bin/bash 
clear
export CUDA_VISIBLE_DEVICES=1


#  ============================================================================================================== 4:task4
for valid_num in 0 1
#1 2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 3 --sample_duration 48 --model_depth 34 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir dep34_crop96_sample224_dura48_aug4_notover_CEL --aug 4 --sample 'none' --task task4 --valid_path '../BenMalData/task/task4/class4_valid5'
done 

#  ============================================================================================================== 4:task4
for valid_num in 0 1
#1 2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 3 --sample_duration 48 --model_depth 34 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir dep34_crop96_sample224_dura48_aug4_notover_CEL --aug 4 --sample 'none' --task task5 --valid_path '../BenMalData/task/task5/class5_valid5'
done 