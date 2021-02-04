#/bin/bash 
clear
export CUDA_VISIBLE_DEVICES=0,1

#  ============================================================================================================== 1:task1
for valid_num in 0
#2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 1 --sample_duration 48 --model_depth 50 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir dep50_crop96_sample224_dura48_aug4_notover_BCE --aug 4 --sample 'not-over' --task task1 --valid_path '../BenMalData/task/task1/class2_valid5'
done 

#  ============================================================================================================== 2:task2
for valid_num in 0
#1 2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 1 --sample_duration 48 --model_depth 50 --batch_size 16 --lr 0.001 --num_valid $valid_num --save_dir dep50_crop96_sample224_dura48_aug4_notover_BCE --aug 4 --sample 'not-over' --task task2 --valid_path '../BenMalData/task/task2/class2_valid5'
done

#  ============================================================================================================== 3:task3
#for valid_num in 0 1
#1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 3 --sample_duration 48 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir dep34_crop96_sample224_dura48_aug4_notover_CEL --aug 4 --sample 'none' --task task3 --valid_path '../BenMalData/task/task3/class3_valid5'
#done 
