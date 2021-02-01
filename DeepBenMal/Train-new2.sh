#！/bin/bash 
clear
export CUDA_VISIBLE_DEVICES=2

#  ==============================================================================================================1
for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_1_time/back_10_32_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 48 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_1_time/back_10_48_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 64 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_1_time/back_10_64_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

#  ==============================================================================================================2
for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_2_time/back_10_32_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 48 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_2_time/back_10_48_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 64 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_2_time/back_10_64_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

#  ==============================================================================================================3
for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_3_time/back_10_32_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 48 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_3_time/back_10_48_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 64 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_3_time/back_10_64_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

#  ==============================================================================================================4
for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 32 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_4_time/back_10_32_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 48 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_4_time/back_10_48_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 

for valid_num in 0 1 2 3 4 5 6 7 8 9
do
python TrainMalBegn-auc-median.py --model 'resnet' --sample_size 64 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 10 --batch_size 32  --lr 0.001 --num_valid $valid_num --save_dir 2019_07_15_4_time/back_10_64_aug4_no_over_AUCH_median --aug 4 --sample 'not-over'  
done 