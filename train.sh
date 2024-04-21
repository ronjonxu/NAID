#!/bin/bash

echo "Start to train the model...."

name="nirnaf"

dataroot='/Data/dataset/Real-NAID/'

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt


python train.py \
    --dataset_name RealNAID --model  nirnaf   --name $name    --dataroot $dataroot  \
    --patch_size 128      --niter 240        --lr_decay_iters 40   --save_imgs True   --lr 3e-4  \
    --batch_size 32       --print_freq 10   --calc_metrics True --gpu_ids 0   --mask_size 1      -j 4   | tee $LOG 
