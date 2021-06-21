ngpu=1
batch_size=${ngpu}
tag=val_dsnet_tracking_slurm_dist

srun -p dsta \
    --job-name=val_dsnet_tracking \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    -w SG-IDC1-10-51-2-74 \
    python -u cfg_train.py \
        --tcp_port 12346 \
        --batch_size ${batch_size} \
        --config cfgs/release/dsnet_tracking.yaml \
        --pretrained_ckpt ./output/train_dsnet_tracking_slurm_dist/ckpt/checkpoint_epoch_2_25.06.pth \
        --tag ${tag} \
        --launcher slurm \
        --fix_semantic_instance \
        --onlyval \
        # --saveval # if you want to save the predictions of the validation set, uncomment this line
