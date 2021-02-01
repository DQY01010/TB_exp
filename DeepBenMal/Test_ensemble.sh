#！/bin/bash 
export CUDA_VISIBLE_DEVICES=3
python TrainMalBegnEnsemble.py --model 'resnet' --n_classes 1 --batch_size 8