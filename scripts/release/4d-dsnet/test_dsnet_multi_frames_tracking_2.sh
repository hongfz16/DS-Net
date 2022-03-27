ngpu=1
tag=test_dsnet_multi_frames_tracking_20000_2

srun -p dsta \
    --job-name=test_dsnet_multi_frames_tracking_20000_2 \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u cfg_train.py \
        --tcp_port 16342 \
        --batch_size ${ngpu} \
        --config cfgs/release/dsnet_multi_frames_tracking_2.yaml \
        --pretrained_ckpt output/train_dsnet_multi_frames_20000_2/ckpt/checkpoint_epoch_5_0.640_0.594_0.648.pth \
        --tag ${tag} \
        --launcher slurm \
        --fix_semantic_instance \
        --onlytest \
