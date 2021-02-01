#！/bin/bash 
clear
export CUDA_VISIBLE_DEVICES=0

#  ============================================================================================================== 1:task1_class2
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir class2_dep10_crop32_aug4_under_FP_test --aug 4 --sample 'under' --task task1 --valid_path '../BenMalData/screenlist/task1/class2_valid5'
#done 

#  ============================================================================================================== 2:task1_class4
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 4 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir class4_dep10_crop32_aug4_CEL --aug 4 --sample 'none' --task task1 --valid_path '../BenMalData/screenlist/task1/class4_valid5'
#done 

#  ============================================================================================================== 3:task2_class2
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir class2_dep10_crop32_aug4_under_FP --aug 4 --sample 'under' --task task2 --valid_path '../BenMalData/screenlist/task2/class2_valid5'
#done 

#  ============================================================================================================== 4:task3_class2
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir class2_dep10_crop32_aug4_under_FP --aug 4 --sample 'under' --task task3 --valid_path '../BenMalData/screenlist/task3/class2_valid5'
#done 

#  ============================================================================================================== 5:task4_class2
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir class2_dep10_crop32_aug4_under_FP --aug 4 --sample 'under' --task task4 --valid_path '../BenMalData/screenlist/task4/class2_valid5'
#done 

#  ============================================================================================================== 6:task2_class4
#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 4 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir class4_dep10_crop32_aug4_CEL --aug 4 --sample 'none' --task task2 --valid_path '../BenMalData/screenlist/task2/class4_valid5'
#done 

#  ============================================================================================================== 7:task3_class4
for valid_num in 0 1 2 3 4
do
python TrainMalBegn.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 4 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir class4_dep10_crop32_aug4_CEL --aug 4 --sample 'none' --task task3 --valid_path '../BenMalData/screenlist/task3/class4_valid5'
done 

#  ============================================================================================================== 8:task4_class4

#for valid_num in 0 1 2 3 4
#do
#python TrainMalBegn.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 4 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir class4_dep10_crop32_aug4_CEL --aug 4 --sample 'none' --task task4 --valid_path '../BenMalData/screenlist/task4/class4_valid5'
#done