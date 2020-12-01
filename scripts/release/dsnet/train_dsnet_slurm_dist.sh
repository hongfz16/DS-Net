ngpu=4
tag=train_dsnet_slurm_dist

srun -p ad_lidar \
    --job-name=train_dsnet \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    python -u cfg_train.py \
        --tcp_port 12345 \
        --batch_size ${ngpu} \
        --config cfgs/release/dsnet.yaml \
        --pretrained_ckpt pretrained_weight/offset_pretrain_pq_0.564.pth \
        --tag ${tag} \
        --launcher slurm \
        --fix_semantic_instance
