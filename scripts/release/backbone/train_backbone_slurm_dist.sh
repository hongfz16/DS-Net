ngpu=2
tag=train_backbone_slurm_dist

srun -p dsta \
    --job-name=train_backbone \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u cfg_train.py \
        --tcp_port 12347 \
        --batch_size ${ngpu} \
        --config cfgs/release/backbone.yaml \
        --pretrained_ckpt pretrained_weight/sem_pretrain.pth \
        --tag ${tag} \
        --launcher slurm
