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

class Siamese_FC(nn.Module):
    def __init__(self, cfg):
        super(Siamese_FC, self).__init__()
        input_dim = cfg.MODEL.TRACKING_HEAD.SIAMESE_INPUT_DIM
        output_dim = cfg.MODEL.TRACKING_HEAD.SIAMESE_OUTPUT_DIM

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        return self.fc(x)
