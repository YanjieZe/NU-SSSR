CUDA_VISIBLE_DEVICES=1,2 python train_gan.py --alg CycleGAN \
                    --fifa \
                    --epoch 201 \
                    --seed 0 \
                    --log_dir logs/cycleGAN_lr_schedule \
                    --wandb \
                    --wandb_key 3961b4db51379fedd20fa7ad0e1b0c5cac1d98ae \
                    --wandb_project DIP \
                    --wandb_group CycleGAN_lr_schedule \
                    --wandb_name 0