#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
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

# python  -m torch.distributed.launch --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29501 train.py
python train.py \
--mode test \
--debug_overFit_train 2 \
--useFeatureAtt 1 \
--accumulation_steps 1 \
--checkpoint_dir checkpoints/pam_aanet_sceneflow \
--batch_size 4 \
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