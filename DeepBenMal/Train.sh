#！/bin/bash 

clear
export CUDA_VISIBLE_DEVICES=3
 
for valid_num in 0 1
do
python TrainMalBegn.py --model 'resnet' --sample_size 64 --epochs 100 --n_classes 1 --sample_duration 16 --model_depth 34 --batch_size 1 --lr 0.001 --num_valid $valid_num
done 
