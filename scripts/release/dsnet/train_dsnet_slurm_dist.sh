ngpu=4
batch_size=8
tag=train_dsnet_slurm_dist_bs_8

srun -p dsta \
    --job-name=train_dsnet \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u cfg_train.py \
        --tcp_port 12340 \
        --batch_size ${batch_size} \
        --config cfgs/release/dsnet.yaml \
        --pretrained_ckpt pretrained_weight/offset_pretrain_pq_0.564.pth \
        --tag ${tag} \
        --launcher slurm \
        --fix_semantic_instance
