# -*- coding:utf-8 -*-
# author: Fangzhou
# @file: pytorch_meanshift.py
# @time: 2020/09/18 16:09

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_cluster import fps
from ..loss.instance_losses import pairwise_distance
from sklearn.neighbors import NearestNeighbors
from utils.common_utils import safe_vis
from utils.config import global_cfg

class PytorchMeanshift(nn.Module):
    def __init__(self, cfg, loss_fn, cluster_fn):
        super(PytorchMeanshift, self).__init__()
        self.bandwidth = cfg.MODEL.MEANSHIFT.BANDWIDTH
        self.iteration = cfg.MODEL.MEANSHIFT.ITERATION
        self.data_mode = cfg.MODEL.MEANSHIFT.DATA_MODE
        self.shift_mode = cfg.MODEL.MEANSHIFT.SHIFT_MODE
        self.down_sample_mode = cfg.MODEL.MEANSHIFT.DOWNSAMPLE_MODE
        self.point_num_th = cfg.MODEL.MEANSHIFT.POINT_NUM_TH
        self.init_size = cfg.MODEL.BACKBONE.INIT_SIZE

        self.meanshift_loss = None
        if 'MEANSHIFT_LOSS' in cfg.MODEL.MEANSHIFT:
            self.meanshift_loss = cfg.MODEL.MEANSHIFT.MEANSHIFT_LOSS
        else:
            if self.data_mode == 'offset':
                self.meanshift_loss = 'offset'
            else:
                raise NotImplementedError

        self.loss_fn = loss_fn
        self.cluster_fn = cluster_fn

        if self.shift_mode  == 'matrix_flat_kernel_bandwidth_weight':
            assert type(self.bandwidth) == list
            self.learnable_bandwidth_weights_layer_list = nn.ModuleList()
            for i in range(self.iteration):
                layer = nn.Sequential(
                    nn.Linear(self.init_size, self.init_size, bias=True),
                    nn.BatchNorm1d(self.init_size),
                    nn.ReLU(),
                    nn.Linear(self.init_size, len(self.bandwidth), bias=True),
                )
                self.learnable_bandwidth_weights_layer_list.append(layer)
        else:
            raise NotImplementedError

    def calc_loss_offset(self, X_list, index, valid, batch_i, batch):
        if len(X_list) == 0:
            return torch.tensor(0.0, requires_grad=True)
        loss_list = []
        if index is not None:
            cpu_index = index.detach().cpu().numpy()
        cur_xyz = batch['pt_cart_xyz'][batch_i][valid]
        cur_offset = batch['pt_offsets'][batch_i][valid]
        if index is not None:
            cur_xyz = cur_xyz[cpu_index]
            cur_offset = cur_offset[cpu_index]
        cur_center = cur_xyz + cur_offset
        safe_vis(cur_center, np.ones(cur_center.shape[0])+1, ignore_zero=False)
        cuda_cur_center = torch.from_numpy(cur_center).cuda()
        for X in X_list:
            diff = X - cuda_cur_center
            dist = torch.sum(torch.abs(diff))
            loss = dist / (X.shape[0] + 1e-6)
            loss_list.append(loss)
        return torch.stack(loss_list).sum()

    def calc_shifted_matrix_flat_kernel_bandwidth_weight(self, X, X_fea, iter_i):
        XT = X.T
        _weights = self.learnable_bandwidth_weights_layer_list[iter_i](X_fea).view(-1, len(self.bandwidth))
        weights = torch.softmax(_weights, dim=1)
        new_X_list = []
        if self.data_mode == 'offset':
            dist = pairwise_distance(X)
        else:
            raise NotImplementedError
        for bandwidth_i in range(len(self.bandwidth)):
            if self.data_mode == 'offset':
                K = (dist <= self.bandwidth[bandwidth_i] ** 2).float()
            else:
                raise NotImplementedError
            D = torch.matmul(K, torch.ones([X.shape[0], 1]).cuda()).view(-1)
            _new_X = torch.matmul(XT, K) / D
            new_X_list.append(_new_X * weights[:, bandwidth_i].view(-1))
        new_X = torch.sum(torch.stack(new_X_list), dim=0) / torch.sum(weights, dim=1).view(-1)
        if torch.isnan(new_X.sum()):
            import pdb; pdb.set_trace()
        if self.data_mode == 'offset':
            return new_X.T, _weights
        else:
            raise NotImplementedError

    def final_cluster(self, final_X, index, data, sampled_data, valid, batch_i, batch):
        # cluster for sampled_data
        sampled_labels = self.cluster_fn([None], [final_X.detach().cpu().numpy()], [None])[0].reshape(-1)

        safe_vis(sampled_data, sampled_labels, ignore_zero=False)

        if index is not None:
            # use NN to assign ins labels to all points in data
            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(sampled_data)
            distances, idxs = nbrs.kneighbors(data)
            labels = sampled_labels[idxs.reshape(-1)]
        else:
            labels = sampled_labels

        safe_vis(data, labels, ignore_zero=False)

        # generate ins labels for all things and stuff points
        clustered_ins_ids = np.zeros(valid.shape[0], dtype=np.int32)
        clustered_ins_ids[valid] = labels

        return clustered_ins_ids

    def down_sample(self, data):
        ratio = (float(self.point_num_th) / data.shape[0]) if data.shape[0] != 0 else 10086
        if ratio >= 1.0:
            return None, None
        # mid_x = data[:, 0].mean()
        # mid_y = data[:, 1].mean()
        # inds = [
        #     (data[:, 0] >= mid_x) & (data[:, 1] >= mid_y),
        #     (data[:, 0] >= mid_x) & (data[:, 1] <  mid_y),
        #     (data[:, 0] <  mid_x) & (data[:, 1] >= mid_y),
        #     (data[:, 0] <  mid_x) & (data[:, 1] <  mid_y),
        # ]
        # for i in inds:
            # cur_i = fps(data[i], torch.zeros(data[i].shape[0]).cuda().long(), ratio=ratio, random_start=False)
            # cur_index_torch = torch.zeros(data[i].shape[0]).cuda().long()
            # cur_index_torch[cur_i] = 1
            # index_torch[i] = cur_index_torch

        # step = data.shape[0] // 4
        # split = [i * step for i in range(5)]
        # split[-1] = data.shape[0]
        # index_torch = torch.zeros(data.shape[0]).cuda().long()
        # for i in range(4):
        #     cur_i = fps(data[split[i]:split[i+1]], torch.zeros(data[split[i]:split[i+1]].shape[0]).cuda().long(), ratio=ratio, random_start=False)
        #     index_torch[cur_i + i*step] = 1

        index_torch = fps(data, torch.zeros(data.shape[0]).cuda().long(), ratio=ratio, random_start=False)
        # index_torch = index_torch == 1
        index = index_torch.detach().cpu().numpy()

        # index = np.random.choice(data.shape[0], int(ratio * data.shape[0]))
        # index_torch = torch.from_numpy(index).cuda()
        return index, index_torch

    def forward(self, xyz_ori_, embedding_ori_, valid_, batch, need_cluster=False):
        # import pdb; pdb.set_trace()
        valid_ = [v!=0 for v in valid_]
        xyz_ = [xyz_ori[valid] for xyz_ori, valid in zip(xyz_ori_, valid_)]
        xyz_torch_ = [torch.from_numpy(xyz).cuda() for xyz in xyz_]
        embedding_ = [embedding_ori[valid] for embedding_ori, valid in zip(embedding_ori_, valid_)]
        if self.down_sample_mode == 'xyz':
            all_index_ = [self.down_sample(xyz) for xyz in xyz_torch_]
            index_np_ = [i[0] for i in all_index_]
            index_ = [i[1] for i in all_index_]
        else:
            raise NotImplementedError
        # sampled_xyz_ = [(xyz[index.detach().cpu().numpy()] if index is not None else xyz) for xyz, index in zip(xyz_, index_)]
        sampled_xyz_ = [(xyz[index] if index is not None else xyz) for xyz, index in zip(xyz_, index_np_)]
        sampled_embedding_ = [(embedding[index] if index is not None else embedding) for embedding, index in zip(embedding_, index_)]

        safe_vis(xyz_ori_[0], batch['pt_ins_labels'][0].reshape(-1), ignore_zero=True)
        if index_[0] is not None:
            safe_vis(sampled_xyz_[0], batch['pt_ins_labels'][0][valid_[0]][index_[0].detach().cpu().numpy()].reshape(-1), ignore_zero=False)
            safe_vis(sampled_embedding_[0].detach().cpu().numpy(), batch['pt_ins_labels'][0][valid_[0]][index_[0].detach().cpu().numpy()].reshape(-1), ignore_zero=False)
        else:
            safe_vis(xyz_[0], batch['pt_ins_labels'][0][valid_[0]].reshape(-1), ignore_zero=False)
            safe_vis(embedding_[0].detach().cpu().numpy(), batch['pt_ins_labels'][0][valid_[0]].reshape(-1), ignore_zero=False)

        batch_size = len(xyz_)
        loss_ = []
        ins_id_ = []
        bandwidth_weight_summary = []
        for batch_i in range(batch_size):
            X = sampled_embedding_[batch_i]
            if X.shape[0] <= 1 and need_cluster:
                ins_id_.append(np.zeros(valid_[batch_i].shape[0], dtype=np.int32))
                loss_.append(torch.tensor(0.0, requires_grad=True).cuda())
                continue
            elif X.shape[0] == 1 and not need_cluster:
                assert index_[batch_i] is None
                valid_[batch_i] = np.zeros(valid_[batch_i].shape, dtype=np.bool)
                X = embedding_ori_[batch_i][valid_[batch_i]]
            iter_X_list = []
            bandwidth_list = []

            learnable_bandwidth = None
            bandwidth_weight = None
            # fixed number of iteration
            if self.shift_mode in ['matrix_flat_kernel_bandwidth_weight']:
                X_fea = batch['ins_fea_list'][batch_i][valid_[batch_i]][index_[batch_i]].reshape(-1, self.init_size)
            for iter_i in range(self.iteration):
                if self.shift_mode == 'matrix_flat_kernel_bandwidth_weight':
                    new_X, bandwidth_weight = self.calc_shifted_matrix_flat_kernel_bandwidth_weight(X, X_fea, iter_i)
                else:
                    raise NotImplementedError
                iter_X_list.append(new_X)
                bandwidth_list.append(bandwidth_weight)
                old_X = X
                X = new_X

                cpu_X = X.detach().cpu().numpy()
                if index_[batch_i] is not None:
                    cpu_X_label = batch['pt_ins_labels'][batch_i][valid_[batch_i]][index_[batch_i].detach().cpu().numpy()].reshape(-1)
                else:
                    cpu_X_label = batch['pt_ins_labels'][batch_i][valid_[batch_i]].reshape(-1)
                safe_vis(cpu_X, cpu_X_label, ignore_zero=False)

                if bandwidth_weight is not None:
                    if index_[batch_i] is not None:
                        cpu_X_sem_label = batch['pt_labs'][batch_i][valid_[batch_i]][index_[batch_i].detach().cpu().numpy()].reshape(-1)
                    else:
                        cpu_X_sem_label = batch['pt_labs'][batch_i][valid_[batch_i]].reshape(-1)
                    uni_cpu_X_sem_label = np.unique(cpu_X_sem_label)
                    bandwidth_weight_softmax = torch.softmax(bandwidth_weight, dim=1).detach().cpu().numpy().reshape(-1, len(self.bandwidth))
                    bandwidth_weight = bandwidth_weight.detach().cpu().numpy().reshape(-1, len(self.bandwidth))
                    bandwidth_weight_summary.append(np.concatenate([old_X.detach().cpu().numpy(), bandwidth_weight_softmax], axis=1))

            bandwidth_weight_summary.append(np.concatenate([new_X.detach().cpu().numpy(), bandwidth_weight_softmax], axis=1))
            if self.data_mode == 'offset':
                if self.meanshift_loss == 'offset':
                    loss = self.calc_loss_offset(iter_X_list, index_[batch_i], valid_[batch_i], batch_i, batch)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            if torch.isnan(loss):
                import pdb; pdb.set_trace()
            loss_.append(loss)

            if need_cluster:
                # final cluster
                if self.down_sample_mode == 'xyz':
                    ins_id = self.final_cluster(X, index_[batch_i], xyz_[batch_i], sampled_xyz_[batch_i], valid_[batch_i], batch_i, batch)
                else:
                    raise NotImplementedError
                ins_id_.append(ins_id)

        return ins_id_, loss_, bandwidth_weight_summary
