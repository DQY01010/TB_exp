#/bin/bash 
clear
export CUDA_VISIBLE_DEVICES=1

#==============resistant&sensitivity：classify2=================
for valid_num in 0 1 2 3 4
#2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 48 --sample_size 64 --epochs 200 --n_classes 3 --sample_duration 16 --model_depth 18 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir resis_dep18Pre_crop48_sample64_dura16_aug8_BCE --aug 8 --sample 'over' --task resis --valid_path '../BenMalData/dataset_split_210301/resisclassify3_valid5_all/' --pretrain_path './pretrained_model/resnet-18-kinetics.pth' --n_finetune_classes 1
done

#==============resistant&sensitivityZclassify2=================
for valid_num in 0 1 2 3 4
#2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 48 --sample_size 64 --epochs 200 --n_classes 3 --sample_duration 16 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir resis_dep34Pre_crop48_sample64_dura16_aug8_BCE_epoch200 --aug 8 --sample 'over' --task resis --valid_path '../BenMalData/dataset_split_210301/resisclassify3_valid5_all/' --pretrain_path './pretrained_model/resnet-34-kinetics.pth' --n_finetune_classes 1
done

