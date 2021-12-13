NPOINT=10000
EPOCH=199

python predict.py --alg SRCNN \
                --data_root data/set5 \
                --seed 0 \
                --predict_epoch $EPOCH \
                --log_dir "logs/srcnn_1000_fourier" \
                --point_num $NPOINT 