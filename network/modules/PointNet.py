import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    def __init__(self, cfg):
        super(PointNet, self).__init__()
        fea_dim = cfg.DATA_CONFIG.DATALOADER.DATA_DIM
        out_pt_fea_dim = cfg.MODEL.VFE.OUT_CHANNEL

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),
            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_pt_fea_dim)
        )

    def forward(self, x):
        return self.PPmodel(x)
