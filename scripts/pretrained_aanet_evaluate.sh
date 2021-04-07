# Evaluate a specific model on Scene Flow test set
export CUDA_VISIBLE_DEVICES=1
 python train.py \
--mode test \
--debug_overFit_train 2 \
--useFeatureAtt 0 \
--checkpoint_dir checkpoints/aanet_sceneflow \
--pretrained_aanet author_pretrained_models/aanet_sceneflow-5aa5a24e.pth \
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