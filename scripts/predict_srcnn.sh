NPOINT=10000
EPOCH=100

python predict.py --alg SRCNN \
                --data_root data/set5 \
                --seed 0 \
                --predict_epoch $EPOCH \
                --log_dir "logs/srcnn_${NPOINT}" \
                --point_num $NPOINT 