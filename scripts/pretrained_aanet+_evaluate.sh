# Evaluate a specific model on Scene Flow test set
export  CUDA_VISIBLE_DEVICES=1
 python train.py \
--mode test \
--debug_overFit_train 2 \
--useFeatureAtt 0 \
--checkpoint_dir checkpoints/aanet+_sceneflow \
--pretrained_aanet author_pretrained_models/aanet+_sceneflow-d3e13ef0.pth \
--batch_size 64 \
--val_batch_size 1 \
--img_height 288 \
--img_width 576 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--milestones 20,30,40,50,60 \
--max_epoch 64 \
--evaluate_only