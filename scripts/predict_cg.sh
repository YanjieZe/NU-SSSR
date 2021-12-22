NPOINT=10000
EPOCH=199
ALG='SRCNN'
# LOGDIR='logs/swinir_denoise_64_20000'
LOGDIR='logs/srcnn_10000'
SAMPLE="random" # or "fourier"

python predict.py --alg $ALG \
                --data_root data/set5 \
                --seed 0 \
                --predict_epoch $EPOCH \
                --log_dir $LOGDIR \
                --point_num $NPOINT \
                --img_width 256 \
                --img_height 256 \
                --img_size 256 \
                --scale 1 \
                --sample_method $SAMPLE