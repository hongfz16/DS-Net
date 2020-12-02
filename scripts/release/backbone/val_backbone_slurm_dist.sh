ngpu=4
tag=val_backebone_slurm_dist

srun -p ad_lidar \
    --job-name=val_backbone \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    python -u cfg_train.py \
        --tcp_port 12345 \
        --batch_size ${ngpu} \
        --config cfgs/release/backbone.yaml \
        --pretrained_ckpt pretrained_weight/offset_pretrain_pq_0.564.pth \
        --tag ${tag} \
        --launcher slurm \
        --onlyval \
        # --saveval # if you want to save the predictions of the validation set, uncomment this line
