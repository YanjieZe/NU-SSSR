export OMP_NUM_THREADS=1
# change num point
python train.py --alg SRCNN \
                --log_dir logs/srcnn_10000 \
                --point_num 10000 \
                --wandb \
                --wandb_project CG \
                --wandb_name 0 \
                --wandb_group SRCNN_10000 \
                --seed 0 \
                --epoch 200

