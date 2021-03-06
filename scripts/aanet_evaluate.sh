#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=1
## Evaluate the best validation model on Scene Flow test set
#CUDA_VISIBLE_DEVICES=1 python  -m torch.distributed.launch  --nproc_per_node 1 train.py \
#--mode test \
#--checkpoint_dir checkpoints/aanet_sceneflow \
#--batch_size 6 \
#--val_batch_size 1 \
#--img_height 288 \
#--img_width 576 \
#--val_img_height 576 \
#--val_img_width 960 \
#--feature_type aanet \
#--feature_pyramid_network \
#--milestones 20,30,40,50,60 \
#--max_epoch 64 \
#--evaluate_only 2>&1 |tee logs/log_test2_aanet_train.txt

# Evaluate the best validation model on Scene Flow test set：the Model is trained using DistributedDataParallel and convert_sync_batchnorm

# by wuzhong, OK to Run
# python  -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29501 train.py
python train.py \
--mode val \
--debug_overFit_train 1_1200 \
--accumulation_steps 1 \
--checkpoint_dir checkpoints/aanet_sceneflow \
--batch_size 6 \
--val_batch_size 1 \
--img_height 288 \
--img_width 576 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type aanet \
--feature_pyramid_network \
--milestones 20,30,40,50,60 \
--max_epoch 64 \
--evaluate_only
#2>&1 |tee logs/log_test_aanet_train.txt

# Evaluate a specific model on Scene Flow test set
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--mode test \
#--checkpoint_dir checkpoints/aanet_sceneflow \
#--pretrained_aanet pretrained/aanet_sceneflow-5aa5a24e.pth \
#--batch_size 64 \
#--val_batch_size 1 \
#--img_height 288 \
#--img_width 576 \
#--val_img_height 576 \
#--val_img_width 960 \
#--feature_type aanet \
#--feature_pyramid_network \
#--milestones 20,30,40,50,60 \
#--max_epoch 64 \
#--evaluate_only

#added by wuzhong
#echo 'Evaluate a specific model on KITTI2015 val set：40 of 200 train images'
#python train.py \
#--mode val \
#--data_dir data/KITTI/kitti_2015/data_scene_flow \
#--dataset_name KITTI2015 \
#--debug_overFit_train 2 \
#--checkpoint_dir checkpoints/aanet_kitti15 \
#--pretrained_aanet author_pretrained_models/aanet_kitti15-fb2a0d23.pth \
#--batch_size 2 \
#--val_batch_size 1 \
#--img_height 336 \
#--img_width 960 \
#--val_img_height 384 \
#--val_img_width 1248 \
#--feature_type aanet \
#--feature_pyramid_network \
#--load_pseudo_gt \
#--highest_loss_only \
#--learning_rate 1e-4 \
#--milestones 400,600,800,900 \
#--max_epoch 1000 \
#--evaluate_only