import random
import torch
import math
import numpy as np

def single_offset_regress_vec(pt_offsets, gt_offsets, valid):
    pt_diff = pt_offsets - gt_offsets   # (N, 3)
    pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
    valid = valid.view(-1).float()
    offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)
    return (offset_norm_loss, )

def pairwise_distance(x: torch.Tensor, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def offset_loss_fun(single_offset_loss_fun):
    def offset_loss(pt_offsets_list, gt_offsets_list, valid_list, gt_semantic_label=None, xyz=None):
        loss_list_list = []
        for i in range(len(pt_offsets_list)):
            loss_list = single_offset_loss_fun(pt_offsets_list[i], gt_offsets_list[i], valid_list[i])
            loss_len = len(loss_list)
            if len(loss_list_list) < loss_len:
                loss_list_list = [[] for j in range(loss_len)]
            for j in range(loss_len):
                loss_list_list[j].append(loss_list[j])
        mean_loss_list = []
        for i in range(len(loss_list_list)):
            mean_loss_list.append(torch.mean(torch.stack(loss_list_list[i])))
        return mean_loss_list
    return offset_loss

offset_loss_regress_vec = offset_loss_fun(single_offset_regress_vec)
