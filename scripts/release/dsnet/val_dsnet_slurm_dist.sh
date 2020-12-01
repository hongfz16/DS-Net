ngpu=4
tag=val_dsnet_slurm_dist

srun -p ad_lidar \
    --job-name=val_dsnet \
    --gres=gpu:${ngpu} \
    --ntasks=${ngpu} \
    --ntasks-per-node=${ngpu} \
    --kill-on-bad-exit=1 \
    python -u cfg_train.py \
        --tcp_port 12345 \
        --batch_size ${ngpu} \
        --config cfgs/release/dsnet.yaml \
        --pretrained_ckpt pretrained_weight/dsnet_pretrain_pq_0.577.pth \
        --tag ${tag} \
        --launcher slurm \
        --fix_semantic_instance \
        --onlyval \
        # --saveval # if you want to save the predictions of the validation set, uncomment this line
