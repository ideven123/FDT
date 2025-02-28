GPUID=6

cd ..

# evaluation of black-box robustness
# remember to first download and put transfer_adv_examples/
# under ../data/
# python eval/eval_bbox.py \
#     --gpu $GPUID \
#     --model-file checkpoints/dverge/seed_0/3_ResNet20_eps_0.07/epoch_200.pth \
#     --folder transfer_adv_examples \
#     --steps 100 \
#     --save-to-csv

# evaluation of white-box robustness
python eval/eval_wbox.py \
   --gpu 0 \
   --model-file /remote-home/ideven/DVERGE_tiny/checkpoint_wrn-res/dverge/seed_0/3_WRN20_eps_0.07/step_10_alpha0.007_fixed_layer_20/fb1.0_epoch200.pth\
   --save-to-csv \
   --model-num 3

python eval/eval_wbox.py \
   --gpu 1 \
   --model-file /remote-home/ideven/DVERGE_tiny/checkpoint_wrn-res/dverge/seed_0/5_WRN20_eps_0.07/step_5_alpha0.007_fixed_layer_20/fb1.0_epoch60.pth\
   --save-to-csv \
   --model-num 8


python eval/eval_wbox.py \
   --gpu 5 \
   --model-file /remote-home/ideven/DVERGE_tiny/checkpoint_icml/dverge/seed_0/3_WRN20_eps_0.05/step_5_alpha0.005_fixed_layer_20/fb1.2_epoch200.pth\
   --save-to-csv \
   --model-num 3


DVERGE/checkpoints_constraint_combinePGD/dverge/seed_0/3_ResNet20_eps_0.08/step_10_alpha0.008_fixed_layer_20/epoch_139.pth
# evaluation of transferability
/data/zwl/zwl/DVERGE_code/checkpoint_zl/dverge/seed_0/3_ResNet20_eps_0.05/step_5_alpha0.005_fixed_layer_20/res10_ta_epoch_200.pth
/data/zwl/zwl/DVERGE_code/checkpoints_zl/baseline_c100/seed_0/3_ResNet20/res_vanilla_cifa100_epoch_200.pth
/data/zwl/zwl/DVERGE_code/checkpoints_zl/dverge/seed_0/3_ResNet20_eps_0.05_fixed_layer_20/c100_deverge_epoch_200.pth
/data/zwl/zwl/DVERGE_code/checkpoints_zl/dverge/seed_0/3_ResNet20_eps_0.05_fixed_layer_20/c100_deverge_epoch_200.pth
python eval/eval_transferability.py \
   --gpu 1 \
   --model-file checkpoints_constraintPGD_maskhigh_0.25/dverge/seed_0/3_ResNet20_eps_0.07_fixed_layer_20/epoch_22.pth \
   --steps 50 \
   --random-start 5 \
   --save-to-file \
   --subset-num 100

# evaluation of diversity
#python eval/eval_diversity.py \
#    --gpu $GPUID \
#    --model-file checkpoints/dverge/seed_0/3_ResNet20_eps_0.07/epoch_200.pth \
#    --save-to-file
 CUDA_VISIBLE_DEVICES=1 python train/train_combine.py --model-num 3 --distill-eps 0.05 --distill-alpha 0.005 --start-from 'baseline' --distill-steps 5