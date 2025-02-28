GPUID=0

cd ..

# DVERGE training
CUDA_VISIBLE_DEVICES=4 python train/train_combine.py --model-num 3 --distill-eps 0.05 --distill-alpha 0.005 --start-from 'baseline' --distill-steps 5

CUDA_VISIBLE_DEVICES=4 python train/train_dverge.py --model-num 3 --distill-eps 0.05 --distill-alpha 0.005 --start-from 'baseline' --distill-steps 5

# python train/train_cross.py --gpu 1  --model-num 3 --eps 0.09 --distill-eps 0.09 # PGDLinf
# python train/train_fixed-distill.py  --model-num 3 --eps 0.09 --distill-eps 0.09 # fixed

CUDA_VISIBLE_DEVICES=5 python train/train_combine.py --model-num 5 --distill-eps 0.07 --distill-alpha 0.007 --start-from 'baseline' --distill-steps 5 --dateset cifar10 --fb 1.0
# Baseline training
CUDA_VISIBLE_DEVICES=4 python train/train_baseline.py --model-num 3

# ADP training
CUDA_VISIBLE_DEVICES=6 python train/train_adv.py  --model-num 3

# GAL training
CUDA_VISIBLE_DEVICES=5 python train/train_gal.py  --model-num 3