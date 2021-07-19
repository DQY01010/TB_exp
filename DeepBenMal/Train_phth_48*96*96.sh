#！/bin/bash 
clear
export CUDA_VISIBLE_DEVICES=1

#  ============================================================================================================== 1:task1_class2
#for valid_num in 0
#1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 1 --sample_duration 48 --model_depth 10 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir class2_dep10_crop96_dura48_aug4_notover_BCE --aug 4 --sample 'not-over' --task task1 --valid_path '../BenMalData/screenlist/task1/class2_valid5'
#done 

#  ============================================================================================================== 2:task1_class4
#for valid_num in 1
#1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 4 --sample_duration 48 --model_depth 10 --batch_size 8  --lr 0.001 --num_valid $valid_num --save_dir class4_dep10_crop96_sample224_dura48_aug4_notover_CEL --aug 4 --sample 'none' --task task1 --valid_path '../BenMalData/screenlist/task1/class4_valid5'
#done 

#  ============================================================================================================== 3:task2_class2
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 96 --epochs 120 --n_classes 1 --sample_duration 48 --model_depth 10 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir class2_dep10_crop96_dura48_aug4_notover_BCE --aug 4 --sample 'not-over' --task task2 --valid_path '../BenMalData/screenlist/task2/class2_valid5'
#done 

#  ============================================================================================================== 4:task3_class2
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 96 --epochs 120 --n_classes 1 --sample_duration 48 --model_depth 10 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir class2_dep10_crop96_dura48_aug4_notover_BCE --aug 4 --sample 'not-over' --task task3 --valid_path '../BenMalData/screenlist/task3/class2_valid5'
#done 

#  ============================================================================================================== 5:task4_class2
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 96 --epochs 120 --n_classes 1 --sample_duration 48 --model_depth 10 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir class2_dep10_crop96_dura48_aug4_notover_BCE --aug 4 --sample 'not-over' --task task4 --valid_path '../BenMalData/screenlist/task4/class2_valid5'
#done 

#  ============================================================================================================== 6:task2_class4
#for valid_num in 0
#1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 4 --sample_duration 48 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir class4_dep34_crop96_sample224_dura48_aug4_notover_CEL --aug 4 --sample 'none' --task task2 --valid_path '../BenMalData/screenlist/task2/class4_valid5'
#done 

#  ============================================================================================================== 7:task3_class4

#for valid_num in 0 1
#do
#python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 4 --sample_duration 32 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir class4_dep34_crop96_sample224_dura32_aug4_notover_CEL --aug 4 --sample 'none' --task task3 --valid_path '../BenMalData/screenlist/task3/class4_valid5'
#done 

#  ============================================================================================================== 8:task4_class4

for valid_num in 0 1
#1 2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 224 --epochs 120 --n_classes 4 --sample_duration 32 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir class4_dep34_crop96_sample224_dura32_aug4_notover_CEL --aug 4 --sample 'none' --task task4 --valid_path '../BenMalData/screenlist/task4/class4_valid5'
done