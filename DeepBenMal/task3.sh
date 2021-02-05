#/bin/bash
clear
export CUDA_VISIBLE_DEVICES=2

for valid_num in 0 1 2 3 4
#1 2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 96 --epochs 120 --n_classes 3 --sample_duration 48 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir dep34Pre_crop96_sample96_dura48_aug8_notover_CEL --aug 8 --sample 'none' --task task3 --valid_path '../BenMalData/task/task3/class3_valid5' --pretrain_path './pretrained_model/resnet-34-kinetics.pth' --n_finetune_classes 3
done

for valid_num in 0 1 2 3 4
#1 2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 96 --epochs 120 --n_classes 3 --sample_duration 48 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir dep34Pre_crop96_sample96_dura48_aug4_notover_CEL_wd1 --aug 4 --sample 'none' --task task3 --valid_path '../BenMalData/task/task3/class3_valid5' --pretrain_path './pretrained_model/resnet-34-kinetics.pth' --n_finetune_classes 3 --weight-decay 1e-1
done

for valid_num in 0 1 2 3 4
#1 2 3 4
do
python TrainMalBegn.py --model 'resnet' --crop_size 96 --sample_size 96 --epochs 120 --n_classes 3 --sample_duration 48 --model_depth 34 --batch_size 16  --lr 0.001 --num_valid $valid_num --save_dir dep34Pre_crop96_sample96_dura48_aug4_notover_CEL_wd2 --aug 4 --sample 'none' --task task3 --valid_path '../BenMalData/task/task3/class3_valid5' --pretrain_path './pretrained_model/resnet-34-kinetics.pth' --n_finetune_classes 3 --weight-decay 1e-2
done

