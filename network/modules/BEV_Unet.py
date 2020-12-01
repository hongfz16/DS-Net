#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_out_fea_cfg(nn.Module):
    def __init__(self,cfg):
        super(UNet_out_fea_cfg, self).__init__()
        n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        dilation = cfg.MODEL.BACKBONE.DILATION
        group_conv = cfg.MODEL.BACKBONE.GROUP_CONV
        input_batch_norm = cfg.MODEL.BACKBONE.INPUT_BATCH_NORM
        dropout = cfg.MODEL.BACKBONE.DROPOUT
        circular_padding = cfg.MODEL.BACKBONE.CIRCULAR_PADDING

        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(1024, 256, circular_padding, group_conv = group_conv)
        self.up2 = up(512, 128, circular_padding, group_conv = group_conv)
        self.up3 = up(256, 64, circular_padding, group_conv = group_conv)
        self.up4 = up(128, 64, circular_padding, group_conv = group_conv)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

class BEV_To_Point_cfg(nn.Module):
    def __init__(self, cfg):
        super(BEV_To_Point_cfg, self).__init__()
        self.fea_dim = cfg.MODEL.BACKBONE.OUT_CHANNEL
        self.use_dim = cfg.MODEL.CLASSIFIER.USE_UNET_CHANNEL
        self.n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        self.dropout = nn.Dropout(p=cfg.MODEL.CLASSIFIER.DROPOUT)
        self.head = outconv(self.fea_dim, self.use_dim * self.n_height)
    
    def forward(self, x, grid_ind):
        grid_fea = self.dropout(x)
        grid_fea = self.head(grid_fea)
        grid_fea = grid_fea.permute(0,2,3,1)
        new_shape = list(grid_fea.size())[:3] + [self.n_height, self.use_dim]
        grid_fea = grid_fea.view(new_shape)
        pt_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_fea_list.append(grid_fea[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        return pt_fea_list

class BEV_Sem_logits_head_cfg(nn.Module):
    def __init__(self, cfg):
        super(BEV_Sem_logits_head_cfg, self).__init__()
        self.n_class = cfg.DATA_CONFIG.NCLASS
        self.n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        self.fea_dim = cfg.MODEL.BACKBONE.OUT_CHANNEL
        self.dropout = nn.Dropout(p=cfg.MODEL.SEM_HEAD.DROPOUT)
        self.head = outconv(self.fea_dim, self.n_class * self.n_height)

    def forward(self, x):
        sem_grid_logits = self.dropout(x)
        sem_grid_logits = self.head(sem_grid_logits)

        sem_grid_logits = sem_grid_logits.permute(0,2,3,1)
        new_shape = list(sem_grid_logits.size())[:3] + [self.n_height,self.n_class]
        sem_grid_logits = sem_grid_logits.view(new_shape)
        sem_grid_logits = sem_grid_logits.permute(0,4,1,2,3)

        return sem_grid_logits

class BEV_Offset_pred_head_cfg(nn.Module):
    def __init__(self, cfg):
        super(BEV_Offset_pred_head_cfg, self).__init__()
        self.n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        self.fea_dim = cfg.MODEL.BACKBONE.OUT_CHANNEL
        self.pt_fea_dim = cfg.MODEL.INS_HEAD.POINT_FEA_CHANNEL
        self.embedding_dim = cfg.MODEL.INS_HEAD.EMBEDDING_CHANNEL

        self.ins_dropout = nn.Dropout(p=cfg.MODEL.INS_HEAD.DROPOUT)
        self.ins_head = outconv(self.fea_dim, self.pt_fea_dim*self.n_height)
        self.offset = nn.Sequential(
            nn.Linear(self.pt_fea_dim, self.pt_fea_dim, bias=True),
            nn.BatchNorm1d(self.pt_fea_dim),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(self.pt_fea_dim, self.embedding_dim, bias=True)

    def forward(self, x, grid_ind):
        ins_grid_fea = self.ins_dropout(x)
        ins_grid_fea = self.ins_head(ins_grid_fea)
        ins_grid_fea = ins_grid_fea.permute(0,2,3,1)
        new_shape = list(ins_grid_fea.size())[:3] + [self.n_height,self.pt_fea_dim]
        ins_grid_fea = ins_grid_fea.view(new_shape)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(ins_grid_fea[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        pt_pred_offsets_list = []
        for pt_ins_fea in pt_ins_fea_list:
            pt_pred_offsets_list.append(self.offset_linear(self.offset(pt_ins_fea)))

        return pt_pred_offsets_list, pt_ins_fea_list

class UNet_out_fea_two_branch_cfg(nn.Module):
    def __init__(self, cfg):
        super(UNet_out_fea_two_branch_cfg, self).__init__()
        n_height = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        dilation = cfg.MODEL.BACKBONE.DILATION
        group_conv = cfg.MODEL.BACKBONE.GROUP_CONV
        input_batch_norm = cfg.MODEL.BACKBONE.INPUT_BATCH_NORM
        dropout = cfg.MODEL.BACKBONE.DROPOUT
        circular_padding = cfg.MODEL.BACKBONE.CIRCULAR_PADDING

        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(1024, 256, circular_padding, group_conv = group_conv)
        self.up2 = up(512, 128, circular_padding, group_conv = group_conv)
        self.up3 = up(256, 64, circular_padding, group_conv = group_conv)
        self.up4 = up(128, 64, circular_padding, group_conv = group_conv)

        self.offset_up1 = up(1024, 256, circular_padding, group_conv = group_conv)
        self.offset_up2 = up(512, 128, circular_padding, group_conv = group_conv)
        self.offset_up3 = up(256, 64, circular_padding, group_conv = group_conv)
        self.offset_up4 = up(128, 64, circular_padding, group_conv = group_conv)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        offset_x = self.offset_up1(x5, x4)
        offset_x = self.offset_up2(offset_x, x3)
        offset_x = self.offset_up3(offset_x, x2)
        offset_x = self.offset_up4(offset_x, x1)

        sem_x = self.up1(x5, x4)
        sem_x = self.up2(sem_x, x3)
        sem_x = self.up3(sem_x, x2)
        sem_x = self.up4(sem_x, x1)

        return sem_x, offset_x

class BEV_Unet(nn.Module):

    def __init__(self,n_class,n_height,dilation = 1,group_conv=False,input_batch_norm = False,dropout = 0.,circular_padding = False):
        super(BEV_Unet, self).__init__()
        self.n_class = n_class
        self.n_height = n_height
        self.network = UNet(n_class*n_height,n_height,dilation,group_conv,input_batch_norm,dropout,circular_padding)

    def forward(self, x):
        x = self.network(x)

        x = x.permute(0,2,3,1)
        new_shape = list(x.size())[:3] + [self.n_height,self.n_class]
        x = x.view(new_shape)
        x = x.permute(0,4,1,2,3)
        return x

class UNet(nn.Module):
    def __init__(self, n_class,n_height,dilation,group_conv,input_batch_norm, dropout,circular_padding):
        super(UNet, self).__init__()
        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(1024, 256, circular_padding, group_conv = group_conv)
        self.up2 = up(512, 128, circular_padding, group_conv = group_conv)
        self.up3 = up(256, 64, circular_padding, group_conv = group_conv)
        self.up4 = up(128, 64, circular_padding, group_conv = group_conv)
        self.dropout = nn.Dropout(p=dropout)
        self.outc = outconv(64, n_class)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(self.dropout(x))
        return x

class BEV_Unet_double_head(nn.Module):

    def __init__(self,n_class,n_height,dilation = 1,group_conv=False,input_batch_norm = False,dropout = 0.,circular_padding = False):
        super(BEV_Unet_double_head, self).__init__()
        self.fea_dim = 64
        self.pt_fea_dim = 16
        self.embedding_dim = 3
        self.n_class = n_class
        self.n_height = n_height
        self.network = UNet_out_fea(n_height,dilation,group_conv,input_batch_norm,dropout,circular_padding)

        self.sem_dropout = nn.Dropout(p=dropout)
        self.sem_head = outconv(self.fea_dim, n_class*n_height)

        self.ins_dropout = nn.Dropout(p=dropout)
        self.ins_head = outconv(self.fea_dim, self.pt_fea_dim*n_height)
        self.offset = nn.Sequential(
            nn.Linear(self.pt_fea_dim, self.pt_fea_dim, bias=True),
            nn.BatchNorm1d(self.pt_fea_dim),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(self.pt_fea_dim, self.embedding_dim, bias=True)

    def forward(self, x, grid_ind):
        grid_fea = self.network(x)
        sem_grid_logits = self.sem_dropout(grid_fea)
        sem_grid_logits = self.sem_head(sem_grid_logits)

        sem_grid_logits = sem_grid_logits.permute(0,2,3,1)
        new_shape = list(sem_grid_logits.size())[:3] + [self.n_height,self.n_class]
        sem_grid_logits = sem_grid_logits.view(new_shape)
        sem_grid_logits = sem_grid_logits.permute(0,4,1,2,3)

        ins_grid_fea = self.ins_dropout(grid_fea)
        ins_grid_fea = self.ins_head(ins_grid_fea)
        ins_grid_fea = ins_grid_fea.permute(0,2,3,1)
        new_shape = list(ins_grid_fea.size())[:3] + [self.n_height,self.pt_fea_dim]
        ins_grid_fea = ins_grid_fea.view(new_shape)
        #ins_grid_fea = ins_grid_fea.permute(0,4,1,2,3)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(ins_grid_fea[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        #print(pt_ins_fea_list[0].size())
        pt_pred_offsets_list = []
        for pt_ins_fea in pt_ins_fea_list:
            pt_pred_offsets_list.append(self.offset_linear(self.offset(pt_ins_fea)))
        #pt_ins_fea = torch.stack(pt_ins_fea_list)
        #print(pt_ins_fea.size())
        #pt_pred_offsets = self.offset(pt_ins_fea)
        #pt_pred_offsets = self.offset_linear(pt_pred_offsets)

        return {"sem_grid_logits": sem_grid_logits, "pt_pred_offsets": pt_pred_offsets_list}

class BEV_Unet_double_branch(nn.Module):
    def __init__(self,n_class,n_height,dilation = 1,group_conv=False,input_batch_norm = False,dropout = 0.,circular_padding = False):
        super(BEV_Unet_double_branch, self).__init__()
        self.fea_dim = 64
        self.pt_fea_dim = 16
        self.embedding_dim = 3
        self.n_class = n_class
        self.n_height = n_height
        self.network = UNet_out_fea_two_branch(n_height,dilation,group_conv,input_batch_norm,dropout,circular_padding)

        self.sem_dropout = nn.Dropout(p=dropout)
        self.sem_head = outconv(self.fea_dim, n_class*n_height)

        self.ins_dropout = nn.Dropout(p=dropout)
        self.ins_head = outconv(self.fea_dim, self.pt_fea_dim*n_height)
        self.offset = nn.Sequential(
            nn.Linear(self.pt_fea_dim, self.pt_fea_dim, bias=True),
            nn.BatchNorm1d(self.pt_fea_dim),
            nn.ReLU()
        )
        self.offset_linear = nn.Linear(self.pt_fea_dim, self.embedding_dim, bias=True)

    def forward(self, x, grid_ind):
        grid_fea, offset_grid_fea = self.network(x)
        sem_grid_logits = self.sem_dropout(grid_fea)
        sem_grid_logits = self.sem_head(sem_grid_logits)

        sem_grid_logits = sem_grid_logits.permute(0,2,3,1)
        new_shape = list(sem_grid_logits.size())[:3] + [self.n_height,self.n_class]
        sem_grid_logits = sem_grid_logits.view(new_shape)
        sem_grid_logits = sem_grid_logits.permute(0,4,1,2,3)

        ins_grid_fea = self.ins_dropout(offset_grid_fea)
        ins_grid_fea = self.ins_head(ins_grid_fea)
        ins_grid_fea = ins_grid_fea.permute(0,2,3,1)
        new_shape = list(ins_grid_fea.size())[:3] + [self.n_height,self.pt_fea_dim]
        ins_grid_fea = ins_grid_fea.view(new_shape)
        #ins_grid_fea = ins_grid_fea.permute(0,4,1,2,3)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):
            pt_ins_fea_list.append(ins_grid_fea[batch_i, grid_ind[batch_i][:,0], grid_ind[batch_i][:,1], grid_ind[batch_i][:,2]])
        #print(pt_ins_fea_list[0].size())
        pt_pred_offsets_list = []
        for pt_ins_fea in pt_ins_fea_list:
            pt_pred_offsets_list.append(self.offset_linear(self.offset(pt_ins_fea)))
        #pt_ins_fea = torch.stack(pt_ins_fea_list)
        #print(pt_ins_fea.size())
        #pt_pred_offsets = self.offset(pt_ins_fea)
        #pt_pred_offsets = self.offset_linear(pt_pred_offsets)

        return {"sem_grid_logits": sem_grid_logits, "pt_pred_offsets": pt_pred_offsets_list}

class UNet_out_fea(nn.Module):
    def __init__(self,n_height,dilation,group_conv,input_batch_norm, dropout,circular_padding):
        super(UNet_out_fea, self).__init__()
        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(1024, 256, circular_padding, group_conv = group_conv)
        self.up2 = up(512, 128, circular_padding, group_conv = group_conv)
        self.up3 = up(256, 64, circular_padding, group_conv = group_conv)
        self.up4 = up(128, 64, circular_padding, group_conv = group_conv)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

class UNet_out_fea_two_branch(nn.Module):
    def __init__(self,n_height,dilation,group_conv,input_batch_norm, dropout,circular_padding):
        super(UNet_out_fea_two_branch, self).__init__()
        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)
        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)
        self.up1 = up(1024, 256, circular_padding, group_conv = group_conv)
        self.up2 = up(512, 128, circular_padding, group_conv = group_conv)
        self.up3 = up(256, 64, circular_padding, group_conv = group_conv)
        self.up4 = up(128, 64, circular_padding, group_conv = group_conv)

        self.offset_up2 = up(512, 128, circular_padding, group_conv = group_conv)
        self.offset_up3 = up(256, 64, circular_padding, group_conv = group_conv)
        self.offset_up4 = up(128, 64, circular_padding, group_conv = group_conv)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)

        offset_x = self.offset_up2(x, x3)
        offset_x = self.offset_up3(offset_x, x2)
        offset_x = self.offset_up4(offset_x, x1)

        sem_x = self.up2(x, x3)
        sem_x = self.up3(sem_x, x2)
        sem_x = self.up4(sem_x, x1)

        return sem_x, offset_x



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1,groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1,groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,group_conv,dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0),groups = min(out_ch,in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0),groups = out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1,0)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        #add circular padding (We implement ring convolution by connecting both ends of matrix via circular padding)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv1(x)
        x = F.pad(x,(1,1,0,0),mode = 'circular')
        x = self.conv2(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch,group_conv = False,dilation = dilation)
            else:
                self.conv = double_conv(in_ch, out_ch,group_conv = False,dilation = dilation)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch,group_conv = group_conv,dilation = dilation)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #TODO: upsample operation is not curcular?
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2,groups = in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch,group_conv = group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch,group_conv = group_conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
