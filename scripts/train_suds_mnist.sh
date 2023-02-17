# Trains suds for the mnist dataset

python suds_main.py \
    --z_dim 128 \
    --channels 1 \
    --batch_size 128 \
    --im_size 32 \
    --k_num 128 \
    --epochs 100 \
    --dataset mnist \
    --savedir models/sanitization \
    --expr_name suds_mnist_128 \
    --no-log