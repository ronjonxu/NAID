#!/bin/bash
echo "Start to test the model...."
name="nirnaf"

dataroot="/Data/dataset/Real-NAID/"  # Modify the path of Real-NAID dataset


device="0"

python test.py \
    --dataset_name  RealNAID    --model nirnaf   --name $name         --dataroot $dataroot  \
    --load_iter    1    --save_imgs True   --calc_metrics True  --gpu_id $device   --mask_size 1  

python metrics3.py  --name $name --dataroot $dataroot --device $device  # test images of three different noise levels independently
python metrics.py  --name $name --dataroot $dataroot --device $device   # test images of three different noise levels together
 
