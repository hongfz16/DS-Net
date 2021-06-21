ngpu=4
batch_size=${ngpu}
tag=train_dsnet_tracking_slurm_dist

srun -p dsta \
    --job-name=train_dsnet_tracking \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u cfg_train.py \
        --tcp_port 14356 \
        --batch_size ${batch_size} \
        --config cfgs/release/dsnet_tracking.yaml \
        --pretrained_ckpt pretrained_weight/dsnet_pretrain_pq_0.577.pth \
        --tag ${tag} \
        --launcher slurm \
        --fix_semantic_instance \
