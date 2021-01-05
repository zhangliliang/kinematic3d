# First train the warmup
CUDA_VISIBLE_DEVICES=0 python scripts/train_rpn_3d.py --config=kitti_3d_warmup_resnet18

# Then train the model with uncertainty
CUDA_VISIBLE_DEVICES=0 python scripts/train_rpn_3d.py --config=kitti_3d_uncertainty_resnet18

# Lastly train the full pose estimation 
CUDA_VISIBLE_DEVICES=0 python scripts/train_pose.py --config=kitti_3d_full_resnet18

CUDA_VISIBLE_DEVICES=0 python scripts/test_kalman.py
