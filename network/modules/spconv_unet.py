# -*- coding:utf-8 -*-
# author: Xinge
# @file: spconv_unet.py
# @time: 2020/06/22 15:01

import time
import numpy as np
import spconv
import torch
import torch.nn.functional as F
from torch import nn

def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                     padding=(0, 1, 1), bias=False, indice_key=indice_key)

def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                     padding=(0, 0, 1), bias=False, indice_key=indice_key)

def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                     padding=(0, 1, 0), bias=False, indice_key=indice_key)

def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                     padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                     padding=(1, 0, 1), bias=False, indice_key=indice_key)

def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=1, bias=False, indice_key=indice_key)



class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key+"bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key+"bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key+"bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key+"bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)
        resA.features = resA.features + shortcut.features

        return resA

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key+"bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key+"bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key+"bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key+"bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        # self.conv4 = conv3x3(out_filters, out_filters, indice_key=indice_key+"bef")
        # self.act4 = nn.LeakyReLU()
        # self.bn4 = nn.BatchNorm1d(out_filters)

        if pooling:
            # self.dropout = nn.Dropout3d(p=dropout_rate)
            if height_pooling:
                # self.pool = spconv.SparseMaxPool3d(kernel_size=2, stride=2)
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                     padding=1, indice_key=indice_key, bias=False)
            else:
                # self.pool = spconv.SparseMaxPool3d(kernel_size=(2,2,1), stride=(2, 2, 1))
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2,2,1),
                     padding=1, indice_key=indice_key, bias=False)
        # else:
        #     self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)

        resA.features = resA.features + shortcut.features

        # resA = self.conv4(resA)
        # resA.features = self.act4(resA.features)
        # resA.features = self.bn4(resA.features)


        if self.pooling:
            # if self.drop_out:
            #     resB = self.dropout(resA.features)
            # else:
            #     resB = resA
            resB = self.pool(resA)

            return resB, resA
        else:
            # if self.drop_out:
            #     resB = self.dropout(resA)
            # else:
            #     resB = resA
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        #self.trans = nn.ConvTranspose2d(in_filters, out_filters, kernel_size, stride=(2, 2), padding=1)
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key+"new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        # self.dropout1 = nn.Dropout3d(p=dropout_rate)
        # self.dropout2 = nn.Dropout3d(p=dropout_rate)

        self.conv1 = conv1x3(out_filters, out_filters,  indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv3x1(out_filters, out_filters,  indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key, bias=False)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        #if upA.shape != skip.shape:
        #    upA = F.pad(upA, (0, 1, 0, 1), mode='replicate')
        upA.features = self.trans_act(upA.features)
        upA.features = self.trans_bn(upA.features)



        ## upsample
        upA = self.up_subm(upA)
        # upA = F.interpolate(upA, size=skip.size()[2:], mode='trilinear', align_corners=True)

        # if self.drop_out:
        #     upA = self.dropout1(upA)
        upA.features = upA.features + skip.features
        # if self.drop_out:
        #     upB = self.dropout2(upB)

        upE = self.conv1(upA)
        upE.features = self.act1(upE.features)
        upE.features = self.bn1(upE.features)



        upE = self.conv2(upE)
        upE.features = self.act2(upE.features)
        upE.features = self.bn2(upE.features)



        upE = self.conv3(upE)
        upE.features = self.act3(upE.features)
        upE.features = self.bn3(upE.features)


        # if self.drop_out:
        #     upE = self.dropout3(upE)

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key+"bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key+"bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key+"bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

        # self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key+"bef")
        # self.act2 = nn.LeakyReLU()
        # self.bn1 = nn.BatchNorm1d(out_filters)
        #
        # self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key+"bef")
        # self.act3 = nn.LeakyReLU()
        # self.bn2 = nn.BatchNorm1d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.bn0(shortcut.features)
        shortcut.features = self.act1(shortcut.features)


        shortcut2 = self.conv1_2(x)
        shortcut2.features = self.bn0_2(shortcut2.features)
        shortcut2.features = self.act1_2(shortcut2.features)


        shortcut3 = self.conv1_3(x)
        shortcut3.features = self.bn0_3(shortcut3.features)
        shortcut3.features = self.act1_3(shortcut3.features)


        # resA = self.conv2(x)
        # resA.features = self.act2(resA.features)
        # resA.features = self.bn1(resA.features)
        #
        # resA = self.conv3(resA)
        # resA.features = self.act3(resA.features)
        # resA.features = self.bn2(resA.features)
        shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut

class Spconv_salsaNet_res_cfg(nn.Module):
    def __init__(self, cfg):
        super(Spconv_salsaNet_res_cfg, self).__init__()

        output_shape = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE
        if 'FEATURE_COMPRESSION' in cfg.MODEL.MODEL_FN:
            num_input_features = cfg.MODEL.MODEL_FN.FEATURE_COMPRESSION
        else:
            num_input_features = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        nclasses = cfg.DATA_CONFIG.NCLASS
        n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        # self.resBlock1 = ResBlock(init_size, init_size, 0.2, pooling=True, height_pooling=True, indice_key="down1")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down5")
        # self.resBlock6 = ResBlock(16 * init_size, 16 * init_size, 0.2, pooling=False, height_pooling=False, indice_key="down6")


        # self.ReconNet = ReconBlock(16 * init_size, 16 * init_size, indice_key="recon")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")
        # self.upBlock4 = UpBlock(4 * init_size, 2 * init_size, indice_key="up4", up_key="down2")
        # self.upBlock5 = UpBlock(2 * init_size, init_size, indice_key="up5", up_key="down1")

        self.ReconNet = ReconBlock(2*init_size, 2*init_size, indice_key="recon")

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down0c, down0b = self.resBlock1(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        # down5b = self.resBlock6(down4c)

        # down6b = self.ReconNet(down5b)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)

        return up0e, up0e

class Spconv_salsaNet_res_merge_cfg(nn.Module):
    def __init__(self, cfg):
        super(Spconv_salsaNet_res_merge_cfg, self).__init__()

        output_shape = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE
        if 'FEATURE_COMPRESSION' in cfg.MODEL.MODEL_FN:
            num_input_features = cfg.MODEL.MODEL_FN.FEATURE_COMPRESSION
        else:
            num_input_features = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        nclasses = cfg.DATA_CONFIG.NCLASS
        n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        # self.resBlock1 = ResBlock(init_size, init_size, 0.2, pooling=True, height_pooling=True, indice_key="down1")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down5")
        # self.resBlock6 = ResBlock(16 * init_size, 16 * init_size, 0.2, pooling=False, height_pooling=False, indice_key="down6")


        # self.ReconNet = ReconBlock(16 * init_size, 16 * init_size, indice_key="recon")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")
        # self.upBlock4 = UpBlock(4 * init_size, 2 * init_size, indice_key="up4", up_key="down2")
        # self.upBlock5 = UpBlock(2 * init_size, init_size, indice_key="up5", up_key="down1")

        self.ReconNet = ReconBlock(2*init_size, 2*init_size, indice_key="recon")

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down0c, down0b = self.resBlock1(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        # down5b = self.resBlock6(down4c)

        # down6b = self.ReconNet(down5b)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)

        return up0e, up0e

class Spconv_sem_logits_head_cfg(nn.Module):
    def __init__(self, cfg):
        super(Spconv_sem_logits_head_cfg, self).__init__()
        output_shape = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE
        if 'FEATURE_COMPRESSION' in cfg.MODEL.MODEL_FN:
            num_input_features = cfg.MODEL.MODEL_FN.FEATURE_COMPRESSION
        else:
            num_input_features = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        nclasses = cfg.DATA_CONFIG.NCLASS
        n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, fea):
        logits = self.logits(fea)
        return logits.dense()

class Spconv_ins_offset_concatxyz_threelayers_head_cfg(nn.Module):
    def __init__(self, cfg):
        super(Spconv_ins_offset_concatxyz_threelayers_head_cfg, self).__init__()
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.pt_fea_dim = 4 * init_size
        self.embedding_dim = cfg.MODEL.INS_HEAD.EMBEDDING_CHANNEL

        self.conv1 = conv3x3(self.pt_fea_dim, self.pt_fea_dim, indice_key='offset_head_conv1')
        self.bn1 = nn.BatchNorm1d(self.pt_fea_dim)
        self.act1 = nn.LeakyReLU()
        self.conv2 = conv3x3(self.pt_fea_dim, 2 * init_size, indice_key='offset_head_conv2')
        self.bn2 = nn.BatchNorm1d(2 * init_size)
        self.act2 = nn.LeakyReLU()
        self.conv3 = conv3x3(2 * init_size, init_size, indice_key='offset_head_conv3')
        self.bn3 = nn.BatchNorm1d(init_size)
        self.act3 = nn.LeakyReLU()

        self.offset = nn.Sequential(
            nn.Linear(init_size+3, init_size, bias=True),
            nn.BatchNorm1d(init_size),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(init_size, self.embedding_dim, bias=True)

    def forward(self, fea, batch, prefix=''):
        fea = self.conv1(fea)
        fea.features = self.act1(self.bn1(fea.features))
        fea = self.conv2(fea)
        fea.features = self.act2(self.bn2(fea.features))
        fea = self.conv3(fea)
        fea.features = self.act3(self.bn3(fea.features))

        grid_ind = batch[prefix + 'grid']
        xyz = batch[prefix + 'pt_cart_xyz']
        fea = fea.dense()
        fea = fea.permute(0, 2, 3, 4, 1)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(fea[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        pt_pred_offsets_list = []
        for batch_i, pt_ins_fea in enumerate(pt_ins_fea_list):
            pt_pred_offsets_list.append(self.offset_linear(self.offset(torch.cat([pt_ins_fea,torch.from_numpy(xyz[batch_i]).cuda()],dim=1))))
        return pt_pred_offsets_list, pt_ins_fea_list

class Spconv_tracking_siamese_head_cfg(nn.Module):
    def __init__(self, cfg):
        super(Spconv_tracking_siamese_head_cfg, self).__init__()
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE
        output_size = cfg.MODEL.TRACKING_HEAD.SIAMESE_INPUT_DIM

        self.pt_fea_dim = 4 * init_size

        self.conv1 = conv3x3(self.pt_fea_dim, self.pt_fea_dim, indice_key='tracking_head_conv1')
        self.bn1 = nn.BatchNorm1d(self.pt_fea_dim)
        self.act1 = nn.LeakyReLU()
        self.conv2 = conv3x3(self.pt_fea_dim, 2 * init_size, indice_key='tracking_head_conv2')
        self.bn2 = nn.BatchNorm1d(2 * init_size)
        self.act2 = nn.LeakyReLU()
        self.conv3 = conv3x3(2 * init_size, init_size, indice_key='tracking_head_conv3')
        self.bn3 = nn.BatchNorm1d(init_size)
        self.act3 = nn.LeakyReLU()

        self.tracking_encoding = nn.Sequential(
            nn.Linear(init_size+3, 64, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size, bias=True),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size, bias=False),
        )

    def forward(self, fea, batch, prefix=''):
        fea = self.conv1(fea)
        fea.features = self.act1(self.bn1(fea.features))
        fea = self.conv2(fea)
        fea.features = self.act2(self.bn2(fea.features))
        fea = self.conv3(fea)
        fea.features = self.act3(self.bn3(fea.features))

        grid_ind = batch[prefix + 'grid']
        xyz = batch[prefix + 'pt_cart_xyz']
        fea = fea.dense()
        fea = fea.permute(0, 2, 3, 4, 1)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(fea[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        pt_pred_tracking_encoding_list = []
        for batch_i, pt_ins_fea in enumerate(pt_ins_fea_list):
            pt_pred_tracking_encoding_list.append(self.tracking_encoding(torch.cat([pt_ins_fea,torch.from_numpy(xyz[batch_i]).cuda()],dim=1)))
        return pt_pred_tracking_encoding_list

class Spconv_tracking_head_cfg(nn.Module):
    def __init__(self, cfg):
        super(Spconv_tracking_head_cfg, self).__init__()
        init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.pt_fea_dim = 4 * init_size

        self.conv1 = conv3x3(self.pt_fea_dim, self.pt_fea_dim, indice_key='tracking_head_conv1')
        self.bn1 = nn.BatchNorm1d(self.pt_fea_dim)
        self.act1 = nn.LeakyReLU()
        self.conv2 = conv3x3(self.pt_fea_dim, 2 * init_size, indice_key='tracking_head_conv2')
        self.bn2 = nn.BatchNorm1d(2 * init_size)
        self.act2 = nn.LeakyReLU()
        self.conv3 = conv3x3(2 * init_size, init_size, indice_key='tracking_head_conv3')
        self.bn3 = nn.BatchNorm1d(init_size)
        self.act3 = nn.LeakyReLU()

        self.tracking_encoding = nn.Sequential(
            nn.Linear(init_size+3, init_size, bias=True),
            nn.BatchNorm1d(init_size),
            nn.ReLU(),
            nn.Linear(init_size, init_size, bias=True),
            nn.BatchNorm1d(init_size),
            nn.ReLU(),
            nn.Linear(init_size, init_size, bias=False),
        )

    def forward(self, fea, batch, prefix=''):
        fea = self.conv1(fea)
        fea.features = self.act1(self.bn1(fea.features))
        fea = self.conv2(fea)
        fea.features = self.act2(self.bn2(fea.features))
        fea = self.conv3(fea)
        fea.features = self.act3(self.bn3(fea.features))

        grid_ind = batch[prefix + 'grid']
        xyz = batch[prefix + 'pt_cart_xyz']
        fea = fea.dense()
        fea = fea.permute(0, 2, 3, 4, 1)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(fea[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        pt_pred_tracking_encoding_list = []
        for batch_i, pt_ins_fea in enumerate(pt_ins_fea_list):
            pt_pred_tracking_encoding_list.append(self.tracking_encoding(torch.cat([pt_ins_fea,torch.from_numpy(xyz[batch_i]).cuda()],dim=1)))
        return pt_pred_tracking_encoding_list

class Spconv_alsaNet_res(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses = 20, n_height = 32, strict=False, init_size=16):
        super(Spconv_alsaNet_res, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        # self.resBlock1 = ResBlock(init_size, init_size, 0.2, pooling=True, height_pooling=True, indice_key="down1")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False, indice_key="down5")
        # self.resBlock6 = ResBlock(16 * init_size, 16 * init_size, 0.2, pooling=False, height_pooling=False, indice_key="down6")


        # self.ReconNet = ReconBlock(16 * init_size, 16 * init_size, indice_key="recon")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")
        # self.upBlock4 = UpBlock(4 * init_size, 2 * init_size, indice_key="up4", up_key="down2")
        # self.upBlock5 = UpBlock(2 * init_size, init_size, indice_key="up5", up_key="down1")

        self.ReconNet = ReconBlock(2*init_size, 2*init_size, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        import pdb
        pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        # down0c, down0b = self.resBlock1(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)
        # down5b = self.resBlock6(down4c)

        # down6b = self.ReconNet(down5b)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)

        # up2e = self.upBlock3(up3e, down2b)
        # up1e = self.upBlock4(up2e, down1b)
        # up0e = self.upBlock5(up1e, down0b)

        # up0e_gap = nn.AdaptiveAvgPool3d((1))(up0e)
        # up0e_gap = F.interpolate(up0e_gap, size=(up0e.size()[2:]), mode='trilinear', align_corners=True)
        # up0e = torch.cat((up0e, up0e_gap), dim=1)

        logits = self.logits(up0e)
        y = logits.dense()
        # y = logits.permute(0, 1, 3, 4, 2)
        return y
