#/bin/bash 
clear
export CUDA_VISIBLE_DEVICES=1

#==============resistant&sensitivity：classify2=================
for valid_num in 0 1 2 3 4
#2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 48 --sample_size 64 --epochs 120 --n_classes 1 --sample_duration 16 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir dep34Pre_crop48_sample64_dura16_aug8_over_FP --aug 8 --sample 'over' --task resis --valid_path '../BenMalData/dataset_split_210301/resistant_valid5_all/' --pretrain_path './pretrained_model/resnet-34-kinetics.pth' --n_finetune_classes 1
done

