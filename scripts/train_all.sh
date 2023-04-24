# train suds mnist
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
    
# train suds cifar
python suds_main.py \
    --z_dim 128 \
    --channels 3 \
    --batch_size 128 \
    --im_size 32 \
    --k_num 128 \
    --epochs 100 \
    --dataset cifar \
    --savedir models/sanitization \
    --expr_name suds_cifar_128 \
    --no-log
    
# train all steghide methods
./train_steg_hide.sh