export OMP_NUM_THREADS=1
ALG='SwinIR'
LOGDIR='logs/swinir_denoise_64_20000'
PNUM=20000
GROUP='SwinIR_denoise_64_20000'

# change num point
CUDA_VISIBLE_DEVICES=4,5,6 python train.py --alg $ALG \
                --log_dir $LOGDIR \
                --point_num $PNUM \
                --batch_size 4 \
                --img_width 64 \
                --img_height 64 \
                --img_size 64 \
                --scale 1 \
                --seed 0 \
                --epoch 200 \
                --wandb \
                --wandb_project CG \
                --wandb_name 0 \
                --wandb_group $GROUP \
                

