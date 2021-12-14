NPOINT=10000
EPOCH=120
ALG='SRCNN'
# ALG='SwinIR'
IMGSIZE=256
# LOGDIR='logs/swinir_denoise_64_20000'
LOGDIR='logs/srcnn_10000'
SAMPLE="random" # or "fourier"

python eval.py --alg $ALG \
                --data_root data/set5 \
                --seed 0 \
                --predict_epoch $EPOCH \
                --log_dir $LOGDIR \
                --point_num $NPOINT \
                --img_width $IMGSIZE \
                --img_height $IMGSIZE \
                --img_size $IMGSIZE \
                --scale 1 \
                --sample_method $SAMPLE