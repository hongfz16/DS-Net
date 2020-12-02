import numpy as np
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

import os

from .dataset import SemKITTI, spherical_dataset, collate_fn_BEV, collate_fn_BEV_test
from utils import common_utils

from utils.config import global_args

__all_voxel_dataset__ =  {
    'Spherical': spherical_dataset
}

class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

def build_dataloader(args, cfg, split='train', logger=None, no_shuffle=False, no_aug=False):
    if logger is not None:
        logger.info("Building dataloader for {} set.".format(split))
    choosen_collate_fn = collate_fn_BEV

    is_training = (split == 'train')
    if cfg.DATA_CONFIG.DATASET_NAME == 'SemanticKitti':
        train_pt_dataset = SemKITTI(
            cfg.DATA_CONFIG.DATASET_PATH + '/sequences/',
            imageset = split,
            return_ref = cfg.DATA_CONFIG.RETURN_REF,
            return_ins = cfg.DATA_CONFIG.RETURN_INS_ID
        )
        train_dataset=__all_voxel_dataset__[cfg.DATA_CONFIG.DATALOADER.VOXEL_TYPE](
            train_pt_dataset,
            grid_size = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE,
            flip_aug = cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.FLIP and is_training and (not no_aug),
            scale_aug = cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.SCALE and is_training and (not no_aug),
            transform_aug= cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.TRANSFORM and is_training and (not no_aug),
            trans_std= cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.TRANSFORM_STD,
            min_rad = -np.pi / 4,
            max_rad = np.pi / 4,
            ignore_label = cfg.DATA_CONFIG.DATALOADER.CONVERT_IGNORE_LABEL,
            rotate_aug = cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.ROTATE and is_training and (not no_aug),
            fixed_volume_space = cfg.DATA_CONFIG.DATALOADER.FIXED_VOLUME_SPACE,
            ## fatal bug!!!
        )
        if logger is not None:
            logger.info("Flip Augmentation: {}".format(cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.FLIP and is_training and (not no_aug)))
            logger.info("Scale Augmentation: {}".format(cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.SCALE and is_training and (not no_aug)))
            logger.info("Transform Augmentation: {}".format(cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.TRANSFORM and is_training and (not no_aug)))
            logger.info("Rotate Augmentation: {}".format(cfg.DATA_CONFIG.DATALOADER.AUGMENTATION.ROTATE and is_training and (not no_aug)))
    else:
        raise NotImplementedError

    if cfg.DIST_TRAIN:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=is_training and (not no_shuffle))
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=False)
    else:
        sampler = None
    train_dataset_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        collate_fn = choosen_collate_fn,
        shuffle = ((sampler is None) and is_training) and (not no_shuffle),
        num_workers = cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
        pin_memory = True,
        drop_last = False,
        sampler = sampler,
        timeout = 0
    )
    if logger is not None:
        logger.info("Shuffle: {}".format(is_training and (not no_shuffle)))

    return train_dataset_loader
