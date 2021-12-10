export OMP_NUM_THREADS=1
# change num point
python train.py --alg SRCNN \
                --log_dir logs/srcnn_1000_fourier \
                --point_num 1000 \
                --wandb \
                --wandb_project CG \
                --wandb_name 0 \
                --wandb_group SRCNN_1000_fourier \
                --seed 0 \
                --epoch 200 

