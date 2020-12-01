ngpu=4
tag=train_backbone_pytorch_dist

python -m torch.distributed.launch --nproc_per_node=${ngpu} cfg_train.py \
    --tcp_port 12345 \
    --batch_size ${ngpu} \
    --config cfgs/release/backbone.yaml \
    --pretrained_ckpt pretrained_weight/sem_pretrain.pth \
    --tag ${tag} \
    --launcher pytorch
