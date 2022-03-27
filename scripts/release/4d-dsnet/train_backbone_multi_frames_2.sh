ngpu=6
tag=train_backbone_multi_frames_2

srun -p dsta \
    --job-name=train_backbone_multi_frames_2 \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u cfg_train.py \
        --tcp_port 12347 \
        --batch_size ${ngpu} \
        --config cfgs/release/backbone_multi_frames_2.yaml \
        --pretrained_ckpt pretrained_weight/offset_pretrain_pq_0.564.pth \
        --tag ${tag} \
        --launcher slurm \
        --nofix \
