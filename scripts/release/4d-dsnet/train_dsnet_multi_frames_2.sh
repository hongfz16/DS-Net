ngpu=4
tag=train_dsnet_multi_frames_20000_2

srun -p dsta \
    --job-name=train_dsnet_multi_frames_20000_2 \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u cfg_train.py \
        --tcp_port 17564 \
        --batch_size ${ngpu} \
        --config cfgs/release/dsnet_multi_frames_2.yaml \
        --pretrained_ckpt output/train_backbone_multi_frames_2/ckpt/checkpoint_epoch_20_0.622_0.572_0.620.pth \
        --tag ${tag} \
        --launcher slurm \
        --fix_semantic_instance
