# -*- coding:utf-8 -*-
# author: Hong Fangzhou
# @file: model_zoo.py
# @time: 2020/09/26 17:05

from .modules import BEV_Unet
from .modules import PointNet
from .modules import spconv_unet
from .modules import pytorch_meanshift
from .loss import instance_losses
from .loss import lovasz_losses
from utils.evaluate_panoptic import init_eval, eval_one_scan_w_fname, printResults, valid_xentropy_ids, class_lut
from utils import clustering
from utils import common_utils
from utils.common_utils import grp_range_torch, parallel_FPS, SemKITTI2train
from scipy.optimize import linear_sum_assignment

import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import numpy as np
import numba as nb
import multiprocessing
from scipy import stats as s
from sklearn.metrics import confusion_matrix as cm
from easydict import EasyDict
import time
import os
import pickle
from sklearn.cluster import MeanShift
from sklearn import manifold, datasets
from scipy import stats as s
from utils import common_utils
from utils.config import global_args

class PolarBaseClass(nn.Module):
    def __init__(self, cfg):
        super(PolarBaseClass, self).__init__()
        self.ignore_label = cfg.DATA_CONFIG.DATALOADER.IGNORE_LABEL
        self.pt_pooling = cfg.MODEL.MODEL_FN.PT_POOLING
        self.max_pt = cfg.MODEL.MODEL_FN.MAX_PT_PER_ENCODE
        self.pt_selection = cfg.MODEL.MODEL_FN.PT_SELECTION
        if 'FEATURE_COMPRESSION' in cfg.MODEL.MODEL_FN.keys():
            self.fea_compre = cfg.MODEL.MODEL_FN.FEATURE_COMPRESSION
        else:
            self.fea_compre = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE[2]
        self.grid_size = cfg.DATA_CONFIG.DATALOADER.GRID_SIZE

        if self.pt_pooling == 'max':
            self.pool_dim = cfg.MODEL.VFE.OUT_CHANNEL

        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU()
            ).cuda()
            self.pt_fea_dim = self.fea_compre

    def voxelize(self, inputs):
        grid_ind = inputs['grid']
        pt_fea = inputs['pt_fea']

        pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pt_fea]
        grid_ind_ten = [torch.from_numpy(i[:, :2]).cuda() for i in grid_ind]

        pt_fea = pt_fea_ten
        xy_ind = grid_ind_ten

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch],(1,0),'constant',value = i_batch))

        cat_pt_fea = torch.cat(pt_fea,dim = 0)
        cat_pt_ind = torch.cat(cat_pt_ind,dim = 0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        cur_dev = pt_fea[0].get_device()
        shuffled_ind = torch.randperm(pt_num,device = cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind,:]
        cat_pt_ind = cat_pt_ind[shuffled_ind,:]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind,return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # subsample pts
        if self.pt_selection == 'random':
            grp_ind = grp_range_torch(unq_cnt,cur_dev)[torch.argsort(torch.argsort(unq_inv))] # convert the array that is in the order of grid to the order of cat_pt_feature
            remain_ind = grp_ind < self.max_pt # randomly sample max_pt points inside a grid
        elif self.pt_selection == 'farthest':
            unq_ind = np.split(np.argsort(unq_inv.detach().cpu().numpy()), np.cumsum(unq_cnt.detach().cpu().numpy()[:-1]))
            remain_ind = np.zeros((pt_num,),dtype = np.bool)
            np_cat_fea = cat_pt_fea.detach().cpu().numpy()[:,:3]
            pool_in = []
            for i_inds in unq_ind:
                if len(i_inds) > self.max_pt:
                    pool_in.append((np_cat_fea[i_inds,:],self.max_pt))
            if len(pool_in) > 0:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                FPS_results = pool.starmap(parallel_FPS, pool_in)
                pool.close()
                pool.join()
            count = 0
            for i_inds in unq_ind:
                if len(i_inds) <= self.max_pt:
                    remain_ind[i_inds] = True
                else:
                    remain_ind[i_inds[FPS_results[count]]] = True
                    count += 1

        cat_pt_fea = cat_pt_fea[remain_ind,:]
        cat_pt_ind = cat_pt_ind[remain_ind,:]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt,max=self.max_pt)

        # process feature
        processed_cat_pt_fea = self.vfe_model(cat_pt_fea)
        #TODO: maybe use pointnet to extract features inside each grid and each grid share the same parameters instead of apply pointnet to global point clouds?
        # This kind of global pointnet is more memory efficient cause otherwise we will have to alloc [480 x 360 x 32 x 64 x C] tensor in order to apply pointnet to each grid

        if self.pt_pooling == 'max':
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0] # choose the max feature for each grid
        else: raise NotImplementedError

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        # stuff pooled data into 4D tensor
        out_data_dim = [len(pt_fea),self.grid_size[0],self.grid_size[1],self.pt_fea_dim]
        out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        out_data[unq[:,0],unq[:,1],unq[:,2],:] = processed_pooled_data
        out_data = out_data.permute(0,3,1,2)

        del pt_fea, xy_ind

        return out_data, grid_ind

    def voxelize_spconv(self, inputs, grid_name='grid', pt_fea_name='pt_fea'):
        grid_ind = inputs[grid_name]
        pt_fea = inputs[pt_fea_name]

        pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in pt_fea]
        grid_ind_ten = [torch.from_numpy(i).cuda() for i in grid_ind]

        pt_fea = pt_fea_ten
        xy_ind = grid_ind_ten

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            cat_pt_ind.append(F.pad(xy_ind[i_batch],(1,0),'constant',value = i_batch))

        cat_pt_fea = torch.cat(pt_fea,dim = 0)
        cat_pt_ind = torch.cat(cat_pt_ind,dim = 0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        cur_dev = pt_fea[0].get_device()
        shuffled_ind = torch.randperm(pt_num,device = cur_dev)
        cat_pt_fea = cat_pt_fea[shuffled_ind,:]
        cat_pt_ind = cat_pt_ind[shuffled_ind,:]

        # unique xy grid index
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind,return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # subsample pts
        if self.pt_selection == 'random':
            grp_ind = grp_range_torch(unq_cnt,cur_dev)[torch.argsort(torch.argsort(unq_inv))] # convert the array that is in the order of grid to the order of cat_pt_feature
            remain_ind = grp_ind < self.max_pt # randomly sample max_pt points inside a grid
        elif self.pt_selection == 'farthest':
            unq_ind = np.split(np.argsort(unq_inv.detach().cpu().numpy()), np.cumsum(unq_cnt.detach().cpu().numpy()[:-1]))
            remain_ind = np.zeros((pt_num,),dtype = np.bool)
            np_cat_fea = cat_pt_fea.detach().cpu().numpy()[:,:3]
            pool_in = []
            for i_inds in unq_ind:
                if len(i_inds) > self.max_pt:
                    pool_in.append((np_cat_fea[i_inds,:],self.max_pt))
            if len(pool_in) > 0:
                pool = multiprocessing.Pool(multiprocessing.cpu_count())
                FPS_results = pool.starmap(parallel_FPS, pool_in)
                pool.close()
                pool.join()
            count = 0
            for i_inds in unq_ind:
                if len(i_inds) <= self.max_pt:
                    remain_ind[i_inds] = True
                else:
                    remain_ind[i_inds[FPS_results[count]]] = True
                    count += 1

        cat_pt_fea = cat_pt_fea[remain_ind,:]
        cat_pt_ind = cat_pt_ind[remain_ind,:]
        unq_inv = unq_inv[remain_ind]
        unq_cnt = torch.clamp(unq_cnt,max=self.max_pt)

        # process feature
        processed_cat_pt_fea = self.vfe_model(cat_pt_fea)
        #TODO: maybe use pointnet to extract features inside each grid and each grid share the same parameters instead of apply pointnet to global point clouds?
        # This kind of global pointnet is more memory efficient cause otherwise we will have to alloc [480 x 360 x 32 x 64 x C] tensor in order to apply pointnet to each grid

        if self.pt_pooling == 'max':
            pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0] # choose the max feature for each grid
        else: raise NotImplementedError

        if self.fea_compre:
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        # stuff pooled data into 4D tensor
        # out_data_dim = [len(pt_fea),self.grid_size[0],self.grid_size[1],self.pt_fea_dim]
        # out_data = torch.zeros(out_data_dim, dtype=torch.float32).to(cur_dev)
        # out_data[unq[:,0],unq[:,1],unq[:,2],:] = processed_pooled_data
        # out_data = out_data.permute(0,3,1,2)

        del pt_fea, xy_ind

        return unq, processed_pooled_data

    def calc_sem_label(self, sem_logits, inputs, need_add_one=True):
        vox_pred_labels = torch.argmax(sem_logits, dim=1)
        vox_pred_labels = vox_pred_labels.cpu().detach().numpy()
        grid_ind = inputs['grid']
        pt_pred_labels = []
        for i in range(len(grid_ind)):
            if need_add_one:
                pt_pred_labels.append(vox_pred_labels[i, grid_ind[i][:, 0], grid_ind[i][:, 1], grid_ind[i][:, 2]] + 1)
            else:
                pt_pred_labels.append(vox_pred_labels[i, grid_ind[i][:, 0], grid_ind[i][:, 1], grid_ind[i][:, 2]])
        return pt_pred_labels

    def calc_sem_label_point_logits(self, sem_logits, inputs, need_add_one=True):
        pts_pred_labels = torch.argmax(sem_logits, dim=1)
        pts_pred_labels = pts_pred_labels.cpu().detach().numpy()
        grid_ind = inputs['grid']
        pt_pred_labels = []
        for i in range(len(grid_ind)):
            if need_add_one:
                pt_pred_labels.append(pts_pred_labels + 1)
            else:
                pt_pred_labels.append(pts_pred_labels)
        return pt_pred_labels

    def update_evaluator(self, evaluator, sem_preds, ins_preds, inputs):
        for i in range(len(sem_preds)):
            eval_one_scan_w_fname(evaluator, inputs['pt_labs'][i].reshape(-1),
                inputs['pt_ins_labels'][i].reshape(-1),
                sem_preds[i], ins_preds[i], inputs['pcd_fname'][i])

    def forward(self, x):
        raise NotImplementedError

class PolarOffset(PolarBaseClass):
    def __init__(self, cfg, need_create_model=True):
        super(PolarOffset, self).__init__(cfg)
        self.ins_loss_name = cfg.MODEL.INS_LOSS
        self.ins_embedding_dim = cfg.MODEL.INS_HEAD.EMBEDDING_CHANNEL
        if not need_create_model:
            return
        self.backbone = getattr(BEV_Unet, cfg.MODEL.BACKBONE.NAME)(cfg)
        self.sem_head = getattr(BEV_Unet, cfg.MODEL.SEM_HEAD.NAME)(cfg)
        self.ins_head = getattr(BEV_Unet, cfg.MODEL.INS_HEAD.NAME)(cfg)
        self.vfe_model = getattr(PointNet, cfg.MODEL.VFE.NAME)(cfg)

        self.ins_loss = getattr(instance_losses, cfg.MODEL.INS_LOSS)
        if cfg.MODEL.SEM_LOSS == 'Lovasz_loss':
            self.sem_loss_lovasz = lovasz_losses.lovasz_softmax
            self.sem_loss = torch.nn.CrossEntropyLoss(ignore_index=cfg.DATA_CONFIG.DATALOADER.IGNORE_LABEL)
        else:
            raise NotImplementedError

        self.cluster_fn_wrapper = getattr(clustering, cfg.MODEL.POST_PROCESSING.CLUSTER_ALGO)
        self.cluster_fn = self.cluster_fn_wrapper(cfg)

        self.merge_func_name = cfg.MODEL.POST_PROCESSING.MERGE_FUNC

    def calc_loss(self, sem_logits, pred_offsets, inputs, need_minus_one=True):
        if need_minus_one:
            vox_label = SemKITTI2train(inputs['vox_label']).type(torch.LongTensor).cuda()
        else:
            vox_label = inputs['vox_label'].type(torch.LongTensor).cuda()

        pt_valid = [torch.from_numpy(i).cuda() for i in inputs['pt_valid']]
        if self.ins_loss_name.find('semantic_centroids') != -1:
            offset_loss_list = self.ins_loss(pred_offsets, inputs['pt_ins_labels'], pt_valid, gt_semantic_label=inputs['pt_labs'])
        elif self.ins_loss_name.find('embedding_contrastive_loss') != -1:
            offset_loss_list = self.ins_loss(pred_offsets, inputs['pt_ins_labels'], pt_valid, gt_semantic_label=inputs['pt_labs'], xyz=inputs['pt_cart_xyz'])
        elif self.ins_loss_name.find('embedding_discriminative') != -1:
            offset_loss_list = self.ins_loss(pred_offsets, inputs['pt_ins_labels'], pt_valid)
        else:
            pt_offsets = [torch.from_numpy(i).cuda() for i in inputs['pt_offsets']]
            offset_loss_list = self.ins_loss(pred_offsets, pt_offsets, pt_valid)

        sem_loss = self.sem_loss_lovasz(torch.nn.functional.softmax(sem_logits), vox_label,ignore=self.ignore_label) + self.sem_loss(sem_logits,vox_label)
        #if self.ins_loss_name == 'embedding_contrastive_loss':
        #    loss = 5 * sem_loss + sum(offset_loss_list)
        #else:
        loss = sem_loss + sum(offset_loss_list)

        ret_dict = {}
        ret_dict['offset_loss_list'] = offset_loss_list
        ret_dict['sem_loss'] = sem_loss
        ret_dict['loss'] = loss

        return ret_dict

    def clustering(self, sem_preds, pred_offsets, inputs):
        grid_ind = inputs['grid']
        pt_cart_xyz = inputs['pt_cart_xyz']
        pt_pred_offsets = [pred_offsets[i].detach().cpu().numpy().reshape(-1, self.ins_embedding_dim) for i in range(len(pred_offsets))]
        pt_pred_valid = []
        for i in range(len(grid_ind)):
            pt_pred_valid.append(np.isin(sem_preds[i], valid_xentropy_ids).reshape(-1))
        pred_ins_ids = self.cluster_fn(pt_cart_xyz, pt_pred_offsets, pt_pred_valid)
        return pred_ins_ids

    def merge_ins_sem(self, sem_preds, pred_ins_ids, logits=None, inputs=None):
        merged_sem_preds = []
        for i in range(len(sem_preds)):
            if self.merge_func_name == 'merge_ins_sem':
                merged_sem_preds.append(common_utils.merge_ins_sem(sem_preds[i], pred_ins_ids[i]))
            elif self.merge_func_name == 'merge_ins_sem_logits_size_based':
                merged_sem_preds.append(common_utils.merge_ins_sem_logits_size_based(sem_preds[i], pred_ins_ids[i], i, logits, inputs))
            elif self.merge_func_name == 'none':
                merged_sem_preds.append(sem_preds[i])
        return merged_sem_preds

    def forward(self, batch, is_test=False, before_merge_evaluator=None, after_merge_evaluator=None, require_cluster=True):
        out_data, grid_ind = self.voxelize(batch)
        sem_fea, ins_fea = self.backbone(out_data)
        sem_logits = self.sem_head(sem_fea)
        pred_offsets, _ = self.ins_head(ins_fea, grid_ind)
        loss_dict = self.calc_loss(sem_logits, pred_offsets, batch)

        if is_test:
            pt_sem_preds = self.calc_sem_label(sem_logits, batch)
            if require_cluster:
                pt_ins_ids_preds = self.clustering(pt_sem_preds, pred_offsets, batch)
            else:
                pt_ins_ids_preds = [np.zeros_like(pt_sem_preds[i]) for i in range(len(pt_sem_preds))]
            if require_cluster:
                merged_sem_preds = self.merge_ins_sem(pt_sem_preds, pt_ins_ids_preds)
            else:
                merged_sem_preds = pt_sem_preds
            if before_merge_evaluator != None:
                self.update_evaluator(before_merge_evaluator, pt_sem_preds, pt_ins_ids_preds, batch)
            if after_merge_evaluator != None:
                self.update_evaluator(after_merge_evaluator, merged_sem_preds, pt_ins_ids_preds, batch)

            loss_dict['sem_preds'] = merged_sem_preds
            loss_dict['ins_preds'] = pt_ins_ids_preds

        return loss_dict

class PolarOffsetSpconv(PolarOffset):
    def __init__(self, cfg):
        super(PolarOffsetSpconv, self).__init__(cfg, need_create_model=False)
        self.backbone = getattr(spconv_unet, cfg.MODEL.BACKBONE.NAME)(cfg)
        self.sem_head = getattr(spconv_unet, cfg.MODEL.SEM_HEAD.NAME)(cfg)
        self.ins_head = getattr(spconv_unet, cfg.MODEL.INS_HEAD.NAME)(cfg)
        self.vfe_model = getattr(PointNet, cfg.MODEL.VFE.NAME)(cfg)

        self.ins_loss = getattr(instance_losses, cfg.MODEL.INS_LOSS)
        if cfg.MODEL.SEM_LOSS == 'Lovasz_loss':
            self.sem_loss_lovasz = lovasz_losses.lovasz_softmax
            if cfg.DATA_CONFIG.DATASET_NAME == 'SemanticKitti' or cfg.DATA_CONFIG.DATASET_NAME == 'SemanticKitti_tracking':
                weights = torch.zeros(20, dtype=torch.float)
                weights[0] = 1.0
                weights[1] = 2.293
                weights[2] = 85.756
                weights[3] = 71.511
                weights[4] = 31.596
                weights[5] = 35.624
                weights[6] = 74.761
                weights[7] = 88.722
                weights[8] = 96.389
                weights[9] = 1.00
                weights[10] = 6.362
                weights[11] = 1.00
                weights[12] = 20.387
                weights[13] = 1.00
                weights[14] = 1.363
                weights[15] = 1.00
                weights[16] = 14.214
                weights[17] = 1.263
                weights[18] = 25.936
                weights[19] = 61.896
            else:
                raise NotImplementedError
            self.sem_loss = torch.nn.CrossEntropyLoss(weight=weights.cuda(), ignore_index=0)
        else:
            raise NotImplementedError

        cluster_fn_wrapper = getattr(clustering, cfg.MODEL.POST_PROCESSING.CLUSTER_ALGO)
        self.cluster_fn = cluster_fn_wrapper(cfg)
        self.is_fix_semantic = False

        self.merge_func_name = cfg.MODEL.POST_PROCESSING.MERGE_FUNC

    def fix_semantic_parameters(self):
        fix_list = [self.backbone, self.sem_head, self.vfe_model, self.fea_compression]
        for mod in fix_list:
            for p in mod.parameters():
                p.requires_grad = False
        self.is_fix_semantic = True

    def forward(self, batch, is_test=False, before_merge_evaluator=None, after_merge_evaluator=None, require_cluster=True, require_merge=True):
        if self.is_fix_semantic:
            with torch.no_grad():
                coor, feature_3d = self.voxelize_spconv(batch)
                sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
                sem_logits = self.sem_head(sem_fea)
        else:
            coor, feature_3d = self.voxelize_spconv(batch)
            sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
            sem_logits = self.sem_head(sem_fea)
        pred_offsets, _ = self.ins_head(ins_fea, batch)
        loss_dict = self.calc_loss(sem_logits, pred_offsets, batch, need_minus_one=False)

        if is_test:
            pt_sem_preds = self.calc_sem_label(sem_logits, batch, need_add_one=False)
            if require_cluster:
                pt_ins_ids_preds = self.clustering(pt_sem_preds, pred_offsets, batch)
            else:
                pt_ins_ids_preds = [np.zeros_like(pt_sem_preds[i]) for i in range(len(pt_sem_preds))]
            if require_merge:
                merged_sem_preds = self.merge_ins_sem(pt_sem_preds, pt_ins_ids_preds, sem_logits, batch)
            else:
                merged_sem_preds = pt_sem_preds
            if before_merge_evaluator != None:
                self.update_evaluator(before_merge_evaluator, pt_sem_preds, pt_ins_ids_preds, batch)
            if after_merge_evaluator != None:
                self.update_evaluator(after_merge_evaluator, merged_sem_preds, pt_ins_ids_preds, batch)

            loss_dict['sem_preds'] = merged_sem_preds
            loss_dict['ins_preds'] = pt_ins_ids_preds
            loss_dict['ins_num'] = np.unique(pt_ins_ids_preds[0]).shape[0]

        return loss_dict

class PolarOffsetSpconvPytorchMeanshift(PolarOffsetSpconv):
    def __init__(self, cfg):
        super(PolarOffsetSpconvPytorchMeanshift, self).__init__(cfg)
        self.pytorch_meanshift = pytorch_meanshift.PytorchMeanshift(cfg, self.ins_loss, self.cluster_fn)
        self.is_fix_semantic_instance = False

    def fix_semantic_instance_parameters(self):
        fix_list = [self.backbone, self.sem_head, self.vfe_model, self.fea_compression, self.ins_head]
        for mod in fix_list:
            for p in mod.parameters():
                p.requires_grad = False
        self.is_fix_semantic_instance = True

    def forward(self, batch, is_test=False, before_merge_evaluator=None, after_merge_evaluator=None, require_cluster=True, require_merge=True):
        if self.is_fix_semantic_instance:
            with torch.no_grad():
                coor, feature_3d = self.voxelize_spconv(batch)
                sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
                sem_logits = self.sem_head(sem_fea)
                pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
        else:
            if self.is_fix_semantic:
                with torch.no_grad():
                    coor, feature_3d = self.voxelize_spconv(batch)
                    sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
                    sem_logits = self.sem_head(sem_fea)
            else:
                coor, feature_3d = self.voxelize_spconv(batch)
                sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
                sem_logits = self.sem_head(sem_fea)
            pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
        loss_dict = self.calc_loss(sem_logits, pred_offsets, batch, need_minus_one=False)
        valid = batch['pt_valid']
        if is_test:
            pt_sem_preds = self.calc_sem_label(sem_logits, batch, need_add_one=False)
            valid = []
            for i in range(len(batch['grid'])):
                valid.append(np.isin(pt_sem_preds[i], valid_xentropy_ids).reshape(-1))
        if self.pytorch_meanshift.data_mode == 'offset':
            embedding = [offset + torch.from_numpy(xyz).cuda() for offset, xyz in zip(pred_offsets, batch['pt_cart_xyz'])]
        else:
            raise NotImplementedError
        batch['ins_fea_list'] = ins_fea_list
        pt_ins_ids_preds, meanshift_loss, bandwidth_weight_summary = self.pytorch_meanshift(batch['pt_cart_xyz'], embedding, valid, batch, need_cluster=is_test)

        loss_dict['bandwidth_weight_summary'] = bandwidth_weight_summary
        loss_dict['meanshift_loss'] = meanshift_loss
        loss_dict['offset_loss_list'] += meanshift_loss
        loss_dict['loss'] += sum(meanshift_loss)

        if is_test:
            if require_cluster:
                merged_sem_preds = self.merge_ins_sem(pt_sem_preds, pt_ins_ids_preds)
            else:
                merged_sem_preds = pt_sem_preds
            if before_merge_evaluator != None:
                self.update_evaluator(before_merge_evaluator, pt_sem_preds, pt_ins_ids_preds, batch)
            if after_merge_evaluator != None:
                self.update_evaluator(after_merge_evaluator, merged_sem_preds, pt_ins_ids_preds, batch)

            loss_dict['sem_preds'] = merged_sem_preds
            loss_dict['ins_preds'] = pt_ins_ids_preds
            loss_dict['ins_num'] = np.unique(pt_ins_ids_preds[0]).shape[0]

        return loss_dict

class PolarOffsetSpconvPytorchMeanshiftTracking(PolarOffsetSpconv):
    def __init__(self, cfg):
        super(PolarOffsetSpconvPytorchMeanshiftTracking, self).__init__(cfg)
        self.cfg = cfg
        self.pytorch_meanshift = pytorch_meanshift.PytorchMeanshift(cfg, self.ins_loss, self.cluster_fn)
        self.tracking_encoder = getattr(spconv_unet, cfg.MODEL.TRACKING_HEAD.NAME)(cfg)
        self.is_fix_semantic_instance = False

        self.is_init = False
        self.before_ins_encoding = None
        self.before_unique_ins_ids = None

    def fix_semantic_instance_parameters(self):
        fix_list = [self.backbone, self.sem_head, self.vfe_model, self.fea_compression, self.ins_head, self.pytorch_meanshift]
        for mod in fix_list:
            for p in mod.parameters():
                p.requires_grad = False
        self.is_fix_semantic_instance = True

    def matching(self, after_ins_ids_gt, after_valid_gt, after_ins_ids_preds, after_valid_preds):
        offset = 2**32

        x_inst_in_cl_mask = after_valid_preds.reshape(-1)
        y_inst_in_cl_mask = after_valid_gt.reshape(-1)
        
        x_inst_in_cl = after_ins_ids_preds.reshape(-1) * x_inst_in_cl_mask.astype(np.int64)
        y_inst_in_cl = after_ins_ids_gt.reshape(-1) * y_inst_in_cl_mask.astype(np.int64)

        unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
        id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
        matched_pred = np.array([False] * unique_pred.shape[0])

        unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
        id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
        matched_gt = np.array([False] * unique_gt.shape[0])

        valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
        offset_combo = x_inst_in_cl[valid_combos] + offset * y_inst_in_cl[valid_combos]
        unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

        gt_labels = unique_combo // offset
        pred_labels = unique_combo % offset
        gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
        pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
        intersections = counts_combo
        unions = gt_areas + pred_areas - intersections
        ious = intersections.astype(np.float) / unions.astype(np.float)

        tp_indexes = ious > 0.5

        return pred_labels[tp_indexes], gt_labels[tp_indexes]

    def tracking(self, before_xyz, before_encoding, before_valid, before_ins_ids,
                 after_xyz, after_encoding, after_valid_gt, after_ins_ids_gt,
                 after_valid_preds, after_ins_ids_preds, min_points=50):
        batch_size = len(before_encoding)
        assigned_ids_list = []
        loss_list = []
        matching_list = []
        for b in range(batch_size):
            before_valid_encoding = before_encoding[b][before_valid[b]]
            before_valid_ins_ids = before_ins_ids[b][before_valid[b]].reshape(-1)
            before_valid_xyz = before_xyz[b][before_valid[b]].reshape(-1, 3)
            before_ins_encoding_list = []
            before_ins_xyz_list = []
            before_unique_ins_ids, before_unique_ins_ids_counts = np.unique(before_valid_ins_ids, return_counts=True)
            before_unique_ins_ids = before_unique_ins_ids[before_unique_ins_ids_counts > min_points].reshape(-1)
            if before_unique_ins_ids.shape[0] == 0:
                # loss_list.append(torch.tensor(0.0, requires_grad=True).cuda())
                loss_list.append((before_valid_encoding - before_valid_encoding).sum())
                assigned_ids_list.append(None)
                continue
            for bu in before_unique_ins_ids:
                before_ins_encoding_list.append(before_valid_encoding[before_valid_ins_ids == bu].mean(0))
                before_ins_xyz_list.append(before_valid_xyz[before_valid_ins_ids == bu])
            before_ins_encoding = torch.cat(before_ins_encoding_list, 0).reshape(before_unique_ins_ids.shape[0], self.cfg.MODEL.BACKBONE.INIT_SIZE)

            after_valid_encoding_preds = after_encoding[b][after_valid_preds[b]]
            after_valid_ins_ids_preds = after_ins_ids_preds[b][after_valid_preds[b]].reshape(-1)
            after_valid_xyz = after_xyz[b][after_valid_preds[b]].reshape(-1, 3)
            after_ins_encoding_list = []
            after_ins_xyz_list = []
            after_unique_ins_ids_preds, after_unique_ins_ids_preds_counts = np.unique(after_valid_ins_ids_preds, return_counts=True)
            after_unique_ins_ids_preds = after_unique_ins_ids_preds[after_unique_ins_ids_preds_counts > min_points].reshape(-1)
            if after_unique_ins_ids_preds.shape[0] == 0:
                # loss_list.append(torch.tensor(0.0, requires_grad=True).cuda())
                loss_list.append((after_valid_encoding_preds - after_valid_encoding_preds).sum())
                assigned_ids_list.append(None)
                continue
            for au in after_unique_ins_ids_preds:
                after_ins_encoding_list.append(after_valid_encoding_preds[after_valid_ins_ids_preds == au].mean(0))
                after_ins_xyz_list.append(after_valid_xyz[after_valid_ins_ids_preds == au])
            after_ins_encoding = torch.cat(after_ins_encoding_list, 0).reshape(after_unique_ins_ids_preds.shape[0], self.cfg.MODEL.BACKBONE.INIT_SIZE)

            before_ins_encoding = before_ins_encoding / (torch.norm(before_ins_encoding, dim=1).reshape(-1, 1) + 1e-8)
            after_ins_encoding = after_ins_encoding / (torch.norm(after_ins_encoding, dim=1).reshape(-1, 1) + 1e-8)

            corr_matrix = torch.matmul(before_ins_encoding, after_ins_encoding.transpose(1, 0)).reshape(
                before_unique_ins_ids.shape[0], after_unique_ins_ids_preds.shape[0]
            )
            # corr_matrix = torch.cat([corr_matrix, torch.zeros(1, after_unique_ins_ids_preds.shape[0]).cuda()], 0)
            # prob_matrix = torch.softmax(corr_matrix, dim=0)
            exp_corr_matrix = torch.exp(corr_matrix)
            exp_corr_matrix = torch.cat([exp_corr_matrix, torch.ones([1, after_unique_ins_ids_preds.shape[0]]).cuda()], 0)
            prob_matrix = exp_corr_matrix / exp_corr_matrix.sum(0)

            assigned_ranks = torch.argmax(prob_matrix, 0)
            before_new_unique_ins_ids = np.concatenate([before_unique_ins_ids, [before_unique_ins_ids.max() + 1]])
            assigned_ids = before_new_unique_ins_ids[assigned_ranks.detach().cpu().numpy()]
            assigned_ids_list.append(assigned_ids)

            pred_labels, gt_labels = self.matching(after_ins_ids_gt[b], after_valid_gt[b], after_ins_ids_preds[b], after_valid_preds[b])

            tp_mask = ((after_unique_ins_ids_preds.reshape(-1, 1) - pred_labels.reshape(1, -1)) == 0).sum(1)
            tp_mask = (tp_mask > 0)
            tp_prob_matrix = prob_matrix[:, tp_mask]

            tp_after_unique_ins_ids_preds = after_unique_ins_ids_preds[tp_mask]
            for i in range(tp_after_unique_ins_ids_preds.shape[0]):
                i_pred_id = tp_after_unique_ins_ids_preds[i]
                i_mask = np.where(pred_labels == i_pred_id)[0][0]
                tp_after_unique_ins_ids_preds[i] = gt_labels[i_mask]

            matching_mask = ((before_unique_ins_ids.reshape(-1, 1) - tp_after_unique_ins_ids_preds.reshape(1, -1)) == 0)
            matching_mask_after = (matching_mask.sum(0) == 0).reshape(1, -1)
            matching_mask_w_new = np.concatenate([matching_mask, matching_mask_after], 0).astype(np.int64)
            assert matching_mask_w_new.shape[0] == tp_prob_matrix.shape[0]
            assert matching_mask_w_new.shape[1] == tp_prob_matrix.shape[1]
            loss = (-torch.log(tp_prob_matrix) * torch.from_numpy(matching_mask_w_new).cuda()).sum()
            loss_list.append(loss)

            matching_list.append({
                'before_xyz': before_xyz[b],
                'after_xyz': after_xyz[b],
                'before_valid': before_valid[b],
                'after_valid_preds': after_valid_preds[b],
                'after_valid_gt': after_valid_gt[b],
                'before_ins_ids': before_ins_ids[b],
                'after_ins_ids_preds': after_ins_ids_preds[b],
                'after_ins_ids_gt': after_ins_ids_gt[b],
                'pred_labels': pred_labels,
                'gt_labels': gt_labels,
                'before_unique_ins_ids': before_unique_ins_ids,
                'after_unique_ins_ids_preds': after_unique_ins_ids_preds,
                'matching_mask_w_new': matching_mask_w_new,
                'assigned_ids': assigned_ids,
                'prob_matrix': prob_matrix,
                'tp_mask': tp_mask,
            })

            if torch.isnan(loss):
                import pdb; pdb.set_trace()

        return assigned_ids_list, loss_list, matching_list

    def tracking_test(self, after_xyz, after_encoding, after_valid_preds, after_ins_ids_preds, other_gt=None, min_points=50):
        batch_size = len(after_xyz)
        assert batch_size == 1
        ins_ids_tracking_list = []
        matching_list = []
        for b in range(batch_size):
            after_valid_encoding_preds = after_encoding[b][after_valid_preds[b]]
            after_valid_ins_ids_preds = after_ins_ids_preds[b][after_valid_preds[b]].reshape(-1)
            after_valid_xyz = after_xyz[b][after_valid_preds[b]].reshape(-1, 3)
            after_ins_encoding_list = []
            after_ins_xyz_list = []
            after_unique_ins_ids_preds, after_unique_ins_ids_preds_counts = np.unique(after_valid_ins_ids_preds, return_counts=True)
            after_unique_ins_ids_preds = after_unique_ins_ids_preds[after_unique_ins_ids_preds_counts > min_points].reshape(-1)
            if after_unique_ins_ids_preds.shape[0] == 0:
                return after_ins_ids_preds, matching_list
            for au in after_unique_ins_ids_preds:
                after_ins_encoding_list.append(after_valid_encoding_preds[after_valid_ins_ids_preds == au].mean(0))
                after_ins_xyz_list.append(after_valid_xyz[after_valid_ins_ids_preds == au])
            after_ins_encoding = torch.cat(after_ins_encoding_list, 0).reshape(after_unique_ins_ids_preds.shape[0], self.cfg.MODEL.BACKBONE.INIT_SIZE)
            after_ins_encoding = after_ins_encoding / (torch.norm(after_ins_encoding, dim=1).reshape(-1, 1) + 1e-8)

            if not self.is_init:
                self.is_init = True
                self.before_ins_encoding = after_ins_encoding
                self.before_unique_ins_ids = after_unique_ins_ids_preds
                return after_ins_ids_preds, matching_list

            corr_matrix = torch.matmul(self.before_ins_encoding, after_ins_encoding.transpose(1, 0)).reshape(
                self.before_unique_ins_ids.shape[0], after_unique_ins_ids_preds.shape[0]
            )
            exp_corr_matrix = torch.exp(corr_matrix)
            exp_corr_matrix = torch.cat([exp_corr_matrix, torch.ones([1, after_unique_ins_ids_preds.shape[0]]).cuda()], 0)
            prob_matrix = exp_corr_matrix / exp_corr_matrix.sum(0)

            assigned_ranks = torch.argmax(prob_matrix, 0) #VPS paper method
            # row_ind, col_ind = linear_sum_assignment(prob_matrix.detach().cpu().numpy())
            # assigned_ranks = row_ind[col_ind.argsort()]
            # VERY WRONG!
            # 1. should try the VPS way
            # 2. hungarian first on the candidates and proposals, then assign new labels to the rest

            before_new_unique_ins_ids = np.concatenate([self.before_unique_ins_ids, [self.before_unique_ins_ids.max() + 1]])
            assigned_ids = before_new_unique_ins_ids[assigned_ranks.detach().cpu().numpy()]
            ins_ids_tracking = np.zeros_like(after_ins_ids_preds[b])
            for i, au in enumerate(after_unique_ins_ids_preds):
                ins_ids_tracking[after_ins_ids_preds[b] == au] = assigned_ids[i]
            ins_ids_tracking_list.append(ins_ids_tracking)

            self.before_ins_encoding = after_ins_encoding
            self.before_unique_ins_ids = after_unique_ins_ids_preds

            matching_list.append({
                'after_xyz': after_xyz[b],
                'after_valid_preds': after_valid_preds[b],
                'after_ins_ids_preds': after_ins_ids_preds[b],
                'after_unique_ins_ids_preds': after_unique_ins_ids_preds,
                'assigned_ids': assigned_ids,
                'prob_matrix': prob_matrix,
            })

        return ins_ids_tracking_list, matching_list

    def forward(self, batch, is_test=False, before_merge_evaluator=None, after_merge_evaluator=None, require_cluster=True, require_merge=True):
        if self.is_fix_semantic_instance:
            with torch.no_grad():
                # coor, feature_3d = self.voxelize_spconv(batch)
                # sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
                # sem_logits = self.sem_head(sem_fea)
                # pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)

                # before_coor, before_feature_3d = self.voxelize_spconv(batch, grid_name='before_grid', pt_fea_name='before_pt_fea')
                # before_sem_fea, before_ins_fea = self.backbone(before_feature_3d, before_coor, len(batch['before_grid']))
                
                if is_test:
                    coor, feature_3d = self.voxelize_spconv(batch)
                    sem_fea, ins_fea = self.backbone(feature_3d, coor, len(batch['grid']))
                    sem_logits = self.sem_head(sem_fea)
                    pred_offsets, ins_fea_list = self.ins_head(ins_fea, batch)
                else:
                    batch['combine_grid'] = batch['grid'] + batch['before_grid']
                    batch['combine_pt_fea'] = batch['pt_fea'] + batch['before_pt_fea']
                    batch['combine_pt_cart_xyz'] = batch['pt_cart_xyz'] + batch['before_pt_cart_xyz']

                    combine_coor, combine_feature_3d = self.voxelize_spconv(batch, grid_name='combine_grid', pt_fea_name='combine_pt_fea')
                    combine_sem_fea, combine_ins_fea = self.backbone(combine_feature_3d, combine_coor, len(batch['combine_grid']))
                    combine_sem_logits = self.sem_head(combine_sem_fea)
                    combine_pred_offsets, combine_ins_fea_list = self.ins_head(combine_ins_fea, batch, prefix='combine_')

                    sem_logits = combine_sem_logits[:len(batch['grid'])]
                    pred_offsets, ins_fea_list = combine_pred_offsets[:len(batch['grid'])], combine_ins_fea_list[:len(batch['grid'])]

        else:
            raise NotImplementedError

        if is_test:
            tracking_encoding = self.tracking_encoder(sem_fea, batch)
        else:
            combine_tracking_encoding = self.tracking_encoder(combine_sem_fea, batch, prefix = 'combine_')
            tracking_encoding = combine_tracking_encoding[:len(batch['grid'])]
            before_tracking_encoding = combine_tracking_encoding[len(batch['before_grid']):]

        loss_dict = self.calc_loss(sem_logits, pred_offsets, batch, need_minus_one=False)

        pt_sem_preds = self.calc_sem_label(sem_logits, batch, need_add_one=False)
        valid = []
        for i in range(len(batch['grid'])):
            valid.append(np.isin(pt_sem_preds[i], valid_xentropy_ids).reshape(-1))
        
        if self.pytorch_meanshift.data_mode == 'offset':
            embedding = [offset + torch.from_numpy(xyz).cuda() for offset, xyz in zip(pred_offsets, batch['pt_cart_xyz'])]
        else:
            raise NotImplementedError
        batch['ins_fea_list'] = ins_fea_list

        if self.is_fix_semantic_instance:
            with torch.no_grad():
                pt_ins_ids_preds, meanshift_loss, bandwidth_weight_summary = self.pytorch_meanshift(batch['pt_cart_xyz'], embedding, valid, batch, need_cluster=True)
        else:
            raise NotImplementedError

        loss_dict['bandwidth_weight_summary'] = bandwidth_weight_summary
        loss_dict['meanshift_loss'] = meanshift_loss
        loss_dict['offset_loss_list'] += meanshift_loss
        loss_dict['loss'] += sum(meanshift_loss)

        merged_sem_preds = self.merge_ins_sem(pt_sem_preds, pt_ins_ids_preds)
        # if before_merge_evaluator != None:
        #     self.update_evaluator(before_merge_evaluator, pt_sem_preds, pt_ins_ids_preds, batch)
        # if after_merge_evaluator != None:
        #     self.update_evaluator(after_merge_evaluator, merged_sem_preds, pt_ins_ids_preds, batch)

        loss_dict['sem_preds'] = merged_sem_preds
        loss_dict['ins_preds'] = pt_ins_ids_preds
        loss_dict['ins_num'] = np.unique(pt_ins_ids_preds[0]).shape[0]

        if is_test:
            other_gt = {
                'after_ins_ids_gt': batch['pt_ins_labels'],
                'after_valid_gt': batch['pt_valid'],
            }
            ins_preds_tracking, matching_list = self.tracking_test(batch['pt_cart_xyz'], tracking_encoding,
                                                                   valid, pt_ins_ids_preds, other_gt=other_gt)
            # loss_dict['ins_preds_tracking'] = ins_preds_tracking
            loss_dict['ins_preds'] = ins_preds_tracking
            loss_dict['matching_list'] = matching_list
        else:
            assigned_ids_list, loss_list, matching_list = self.tracking(batch['before_pt_cart_xyz'], before_tracking_encoding,
                                                                        batch['before_pt_valid'], batch['before_pt_ins_labels'],
                                                                        batch['pt_cart_xyz'], tracking_encoding,
                                                                        batch['pt_valid'], batch['pt_ins_labels'],
                                                                        valid, pt_ins_ids_preds)
            loss_dict['assigned_ids_list'] = assigned_ids_list
            loss_dict['tracking_loss'] = loss_list
            loss_dict['matching_list'] = matching_list
        return loss_dict

class PolarOffsetSpconvPytorchMeanshiftTrackingSiamese(PolarOffsetSpconvPytorchMeanshiftTracking):
    def __init__(self, cfg):
        super(PolarOffsetSpconvPytorchMeanshiftTrackingSiamese, self).__init__(cfg)
        self.siamese_fc = PointNet.Siamese_FC(cfg)

    def tracking(self, before_xyz, before_encoding, before_valid, before_ins_ids,
                 after_xyz, after_encoding, after_valid_gt, after_ins_ids_gt,
                 after_valid_preds, after_ins_ids_preds, min_points=5):
        batch_size = len(before_encoding)
        assigned_ids_list = []
        loss_list = []
        matching_list = []
        for b in range(batch_size):
            before_valid_encoding = before_encoding[b][before_valid[b]]
            before_valid_ins_ids = before_ins_ids[b][before_valid[b]].reshape(-1)
            before_valid_xyz = before_xyz[b][before_valid[b]].reshape(-1, 3)
            before_ins_encoding_list = []
            before_unique_ins_ids = np.unique(before_valid_ins_ids).reshape(-1)
            if before_unique_ins_ids.shape[0] == 0:
                loss_list.append((before_valid_encoding - before_valid_encoding).sum())
                assigned_ids_list.append(None)
                continue
            for bu in before_unique_ins_ids:
                before_ins_encoding_list.append(before_valid_encoding[before_valid_ins_ids == bu].max(0)[0])
            before_ins_encoding = torch.cat(before_ins_encoding_list, 0).reshape(before_unique_ins_ids.shape[0], self.cfg.MODEL.TRACKING_HEAD.SIAMESE_INPUT_DIM)
            before_ins_encoding = self.siamese_fc(before_ins_encoding).reshape(before_unique_ins_ids.shape[0], self.cfg.MODEL.TRACKING_HEAD.SIAMESE_OUTPUT_DIM)

            after_valid_encoding_preds = after_encoding[b][after_valid_preds[b]]
            after_valid_ins_ids_preds = after_ins_ids_preds[b][after_valid_preds[b]].reshape(-1)
            after_valid_xyz = after_xyz[b][after_valid_preds[b]].reshape(-1, 3)
            after_ins_encoding_list = []
            after_unique_ins_ids_preds, after_unique_ins_ids_preds_counts = np.unique(after_valid_ins_ids_preds, return_counts=True)
            after_unique_ins_ids_preds = after_unique_ins_ids_preds[after_unique_ins_ids_preds_counts > min_points].reshape(-1)
            if after_unique_ins_ids_preds.shape[0] == 0:
                loss_list.append((after_valid_encoding_preds - after_valid_encoding_preds).sum())
                assigned_ids_list.append(None)
                continue
            for au in after_unique_ins_ids_preds:
                after_ins_encoding_list.append(after_valid_encoding_preds[after_valid_ins_ids_preds == au].max(0)[0])
            after_ins_encoding = torch.cat(after_ins_encoding_list, 0).reshape(after_unique_ins_ids_preds.shape[0], self.cfg.MODEL.TRACKING_HEAD.SIAMESE_INPUT_DIM)
            after_ins_encoding = self.siamese_fc(after_ins_encoding).reshape(after_unique_ins_ids_preds.shape[0], self.cfg.MODEL.TRACKING_HEAD.SIAMESE_OUTPUT_DIM)

            corr_matrix = torch.matmul(before_ins_encoding, after_ins_encoding.transpose(1, 0)).reshape(
                before_unique_ins_ids.shape[0], after_unique_ins_ids_preds.shape[0]
            )
            corr_matrix = torch.cat([corr_matrix, torch.zeros(1, after_unique_ins_ids_preds.shape[0]).cuda()], 0)
            prob_matrix = torch.softmax(corr_matrix, dim=0)

            assigned_ranks = torch.argmax(prob_matrix, 0)
            before_new_unique_ins_ids = np.concatenate([before_unique_ins_ids, [before_unique_ins_ids.max() + 1]])
            assigned_ids = before_new_unique_ins_ids[assigned_ranks.detach().cpu().numpy()]
            assigned_ids_list.append(assigned_ids)

            pred_labels, gt_labels = self.matching(after_ins_ids_gt[b], after_valid_gt[b], after_ins_ids_preds[b], after_valid_preds[b])

            tp_mask = ((after_unique_ins_ids_preds.reshape(-1, 1) - pred_labels.reshape(1, -1)) == 0).sum(1)
            tp_mask = (tp_mask > 0)
            # tp_prob_matrix = prob_matrix[:, tp_mask]
            tp_corr_matrix = corr_matrix[:, tp_mask]

            tp_after_unique_ins_ids_preds = after_unique_ins_ids_preds[tp_mask]
            for i in range(tp_after_unique_ins_ids_preds.shape[0]):
                i_pred_id = tp_after_unique_ins_ids_preds[i]
                i_mask = np.where(pred_labels == i_pred_id)[0][0]
                tp_after_unique_ins_ids_preds[i] = gt_labels[i_mask]

            matching_mask = ((before_unique_ins_ids.reshape(-1, 1) - tp_after_unique_ins_ids_preds.reshape(1, -1)) == 0)
            matching_mask_after = (matching_mask.sum(0) == 0).reshape(1, -1)
            matching_mask_w_new = np.concatenate([matching_mask, matching_mask_after], 0).astype(np.int64)
            # assert matching_mask_w_new.shape[0] == tp_prob_matrix.shape[0]
            # assert matching_mask_w_new.shape[1] == tp_prob_matrix.shape[1]
            assert matching_mask_w_new.shape[0] == tp_corr_matrix.shape[0]
            assert matching_mask_w_new.shape[1] == tp_corr_matrix.shape[1]
            # loss_matrix = -torch.log(tp_prob_matrix) * torch.from_numpy(matching_mask_w_new).cuda()
            if torch.any(torch.isnan(tp_corr_matrix)):
                import pdb; pdb.set_trace()
            if torch.any(torch.isinf(tp_corr_matrix)):
                import pdb; pdb.set_trace()
            loss_matrix = -torch.nn.functional.log_softmax(tp_corr_matrix, dim=0) * torch.from_numpy(matching_mask_w_new).cuda()
            loss_matrix[torch.where(torch.isnan(loss_matrix))] = 0
            loss = loss_matrix.sum()
            loss_list.append(loss)

            matching_list.append({
                'before_xyz': before_xyz[b],
                'after_xyz': after_xyz[b],
                'before_valid': before_valid[b],
                'after_valid_preds': after_valid_preds[b],
                'after_valid_gt': after_valid_gt[b],
                'before_ins_ids': before_ins_ids[b],
                'after_ins_ids_preds': after_ins_ids_preds[b],
                'after_ins_ids_gt': after_ins_ids_gt[b],
                'pred_labels': pred_labels,
                'gt_labels': gt_labels,
                'before_unique_ins_ids': before_unique_ins_ids,
                'after_unique_ins_ids_preds': after_unique_ins_ids_preds,
                'matching_mask_w_new': matching_mask_w_new,
                'assigned_ids': assigned_ids,
                'prob_matrix': prob_matrix,
                'tp_mask': tp_mask,
            })

            if torch.isnan(loss):
                import pdb; pdb.set_trace()

        return assigned_ids_list, loss_list, matching_list

    def tracking_test(self, after_xyz, after_encoding, after_valid_preds, after_ins_ids_preds, other_gt=None, min_points=5):
        batch_size = len(after_xyz)
        assert batch_size == 1
        ins_ids_tracking_list = []
        matching_list = []
        for b in range(batch_size):
            after_valid_encoding_preds = after_encoding[b][after_valid_preds[b]]
            after_valid_ins_ids_preds = after_ins_ids_preds[b][after_valid_preds[b]].reshape(-1)
            after_valid_xyz = after_xyz[b][after_valid_preds[b]].reshape(-1, 3)
            after_ins_encoding_list = []
            after_unique_ins_ids_preds, after_unique_ins_ids_preds_counts = np.unique(after_valid_ins_ids_preds, return_counts=True)
            after_unique_ins_ids_preds = after_unique_ins_ids_preds[after_unique_ins_ids_preds_counts > min_points].reshape(-1)
            if after_unique_ins_ids_preds.shape[0] == 0:
                return after_ins_ids_preds, matching_list
            for au in after_unique_ins_ids_preds:
                after_ins_encoding_list.append(after_valid_encoding_preds[after_valid_ins_ids_preds == au].max(0)[0])
            after_ins_encoding = torch.cat(after_ins_encoding_list, 0).reshape(after_unique_ins_ids_preds.shape[0], self.cfg.MODEL.TRACKING_HEAD.SIAMESE_INPUT_DIM)
            after_ins_encoding = self.siamese_fc(after_ins_encoding).reshape(after_unique_ins_ids_preds.shape[0], self.cfg.MODEL.TRACKING_HEAD.SIAMESE_OUTPUT_DIM)

            if not self.is_init:
                self.is_init = True
                self.before_ins_encoding = after_ins_encoding
                self.before_unique_ins_ids = after_unique_ins_ids_preds
                return after_ins_ids_preds, matching_list

            corr_matrix = torch.matmul(self.before_ins_encoding, after_ins_encoding.transpose(1, 0)).reshape(
                self.before_unique_ins_ids.shape[0], after_unique_ins_ids_preds.shape[0]
            )
            corr_matrix = torch.cat([corr_matrix, torch.zeros(1, after_unique_ins_ids_preds.shape[0]).cuda()], 0)
            prob_matrix = torch.softmax(corr_matrix, dim=0)

            # import pdb; pdb.set_trace()

            assigned_ranks = torch.argmax(prob_matrix, 0)

            # _assigned_ranks = torch.argmax(prob_matrix, 0)
            # assigned_ranks = torch.zeros_like(_assigned_ranks) - 1
            # new_ins_ind = self.before_unique_ins_ids.shape[0]
            # is_new_ins_ind = (_assigned_ranks == new_ins_ind)
            # assigned_ranks[is_new_ins_ind] = new_ins_ind
            # cost_matrix = -prob_matrix[:-1, ~is_new_ins_ind].detach().cpu().numpy()
            # row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # old_assigned_ranks = torch.zeros_like(assigned_ranks[~is_new_ins_ind]) - 1
            # for r, c in zip(row_ind, col_ind):
            #     old_assigned_ranks[c] = r
            # old_assigned_ranks[old_assigned_ranks == -1] = new_ins_ind
            # assigned_ranks[~is_new_ins_ind] = old_assigned_ranks

            new_label_start = self.before_unique_ins_ids.max() + 1
            before_new_unique_ins_ids = np.concatenate([self.before_unique_ins_ids, [new_label_start]])
            assigned_ids = before_new_unique_ins_ids[assigned_ranks.detach().cpu().numpy()]
            flag = False
            current_id = new_label_start + 1
            for i, id in enumerate(assigned_ids):
                if id == new_label_start:
                    if flag:
                        assigned_ids[i] = current_id
                        current_id += 1
                    else:
                        flag = True
            ins_ids_tracking = np.zeros_like(after_ins_ids_preds[b])
            for i, au in enumerate(after_unique_ins_ids_preds):
                ins_ids_tracking[after_ins_ids_preds[b] == au] = assigned_ids[i]
            ins_ids_tracking_list.append(ins_ids_tracking)

            self.before_ins_encoding = after_ins_encoding
            self.before_unique_ins_ids = after_unique_ins_ids_preds

            matching_list.append({
                'after_xyz': after_xyz[b],
                'after_valid_preds': after_valid_preds[b],
                'after_ins_ids_preds': after_ins_ids_preds[b],
                'after_unique_ins_ids_preds': after_unique_ins_ids_preds,
                'assigned_ids': assigned_ids,
                'prob_matrix': prob_matrix,
            })

        return ins_ids_tracking_list, matching_list

class PolarOffsetSpconvPytorchMeanshiftTrackingSiameseWithGeoClue(PolarOffsetSpconvPytorchMeanshiftTrackingSiamese):
    def __init__(self, cfg):
        super(PolarOffsetSpconvPytorchMeanshiftTrackingSiameseWithGeoClue, self).__init__(cfg)
        self.before_centers = None

    def tracking_test(self, after_xyz, after_encoding, after_valid_preds, after_ins_ids_preds, other_gt=None, min_points=50):
        batch_size = len(after_xyz)
        assert batch_size == 1
        ins_ids_tracking_list = []
        matching_list = []
        for b in range(batch_size):
            after_valid_encoding_preds = after_encoding[b][after_valid_preds[b]]
            after_valid_ins_ids_preds = after_ins_ids_preds[b][after_valid_preds[b]].reshape(-1)
            after_valid_xyz = after_xyz[b][after_valid_preds[b]].reshape(-1, 3)
            after_ins_encoding_list = []
            after_centers_list = []
            after_unique_ins_ids_preds, after_unique_ins_ids_preds_counts = np.unique(after_valid_ins_ids_preds, return_counts=True)
            after_unique_ins_ids_preds = after_unique_ins_ids_preds[after_unique_ins_ids_preds_counts > min_points].reshape(-1)
            if after_unique_ins_ids_preds.shape[0] == 0:
                return after_ins_ids_preds, matching_list
            for au in after_unique_ins_ids_preds:
                after_ins_encoding_list.append(after_valid_encoding_preds[after_valid_ins_ids_preds == au].max(0)[0])
                after_centers_list.append(after_valid_xyz[after_valid_ins_ids_preds == au].mean(0))
            after_ins_encoding = torch.cat(after_ins_encoding_list, 0).reshape(after_unique_ins_ids_preds.shape[0], self.cfg.MODEL.TRACKING_HEAD.SIAMESE_INPUT_DIM)
            after_ins_encoding = self.siamese_fc(after_ins_encoding).reshape(after_unique_ins_ids_preds.shape[0], self.cfg.MODEL.TRACKING_HEAD.SIAMESE_OUTPUT_DIM)
            after_centers = np.concatenate(after_centers_list, 0).reshape(after_unique_ins_ids_preds.shape[0], 3)
            after_centers = torch.from_numpy(after_centers).cuda()

            if not self.is_init:
                self.is_init = True
                self.before_ins_encoding = after_ins_encoding
                self.before_unique_ins_ids = after_unique_ins_ids_preds
                self.before_centers = after_centers
                pred_labels, gt_labels = self.matching(other_gt['after_ins_ids_gt'][b], other_gt['after_valid_gt'][b], after_ins_ids_preds[b], after_valid_preds[b])
                pred2gt_dict = {}
                for p, g in zip(pred_labels, gt_labels):
                    pred2gt_dict[p] = g
                self.before_pred2gt_dict = pred2gt_dict
                return after_ins_ids_preds, matching_list

            corr_matrix = torch.matmul(self.before_ins_encoding, after_ins_encoding.transpose(1, 0)).reshape(
                self.before_unique_ins_ids.shape[0], after_unique_ins_ids_preds.shape[0]
            )
            corr_matrix = torch.cat([corr_matrix, torch.zeros(1, after_unique_ins_ids_preds.shape[0]).cuda()], 0)
            prob_matrix = torch.softmax(corr_matrix, dim=0)

            dist_matrix = self.before_centers.reshape(-1, 1, 3) - after_centers.reshape(1, -1, 3)
            dist_matrix = dist_matrix.pow(2).sum(-1).sqrt()

            #### COMBINE PROB_MATRIX & DISTANCE MATRIX ####
            # dist_softmax_matrix = torch.softmax(dist_matrix, dim=0)
            # prob_matrix[:-1, :] -= dist_softmax_matrix
            # dist_norm_matrix = dist_matrix / dist_matrix.max()
            # prob_matrix[:-1, :] -= dist_norm_matrix
            ########

            #### MASK OUT IMPOSSIBLE PAIRS ####
            dist_impossible_mask = (dist_matrix > 2)
            prob_matrix[:-1, :][dist_impossible_mask] = -10086
            ########

            # import pdb; pdb.set_trace()

            _assigned_ranks = torch.argmax(prob_matrix, 0)
            assigned_ranks = torch.zeros_like(_assigned_ranks) - 1
            new_ins_ind = self.before_unique_ins_ids.shape[0]
            is_new_ins_ind = (_assigned_ranks == new_ins_ind)
            assigned_ranks[is_new_ins_ind] = new_ins_ind
            cost_matrix = -prob_matrix[:-1, ~is_new_ins_ind].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            old_assigned_ranks = torch.zeros_like(assigned_ranks[~is_new_ins_ind]) - 1
            for r, c in zip(row_ind, col_ind):
                old_assigned_ranks[c] = r
            old_assigned_ranks[old_assigned_ranks == -1] = new_ins_ind
            assigned_ranks[~is_new_ins_ind] = old_assigned_ranks

            new_label_start = self.before_unique_ins_ids.max() + 1
            before_new_unique_ins_ids = np.concatenate([self.before_unique_ins_ids, [new_label_start]])
            assigned_ids = before_new_unique_ins_ids[assigned_ranks.detach().cpu().numpy()]
            flag = False
            current_id = new_label_start + 1
            for i, id in enumerate(assigned_ids):
                if id == new_label_start:
                    if flag:
                        assigned_ids[i] = current_id
                        current_id += 1
                    else:
                        flag = True
            ins_ids_tracking = np.zeros_like(after_ins_ids_preds[b])
            for i, au in enumerate(after_unique_ins_ids_preds):
                ins_ids_tracking[after_ins_ids_preds[b] == au] = assigned_ids[i]
            ins_ids_tracking_list.append(ins_ids_tracking)

            pred_labels, gt_labels = self.matching(other_gt['after_ins_ids_gt'][b], other_gt['after_valid_gt'][b], ins_ids_tracking, ins_ids_tracking>0)
            pred2gt_dict = {}
            for p, g in zip(pred_labels, gt_labels):
                pred2gt_dict[p] = g
            
            before_tp_count = len(self.before_pred2gt_dict.keys())
            match_tp_count = 0
            current_tp_count = len(pred2gt_dict.keys())
            current_ins_count = np.unique(ins_ids_tracking).shape[0]
            for bk in self.before_pred2gt_dict.keys():
                if bk in pred2gt_dict:
                    if self.before_pred2gt_dict[bk] == pred2gt_dict[bk]:
                        match_tp_count += 1
            
            print("M/B: {}/{}; M/C: {}/{}; IC: {}".format(match_tp_count, before_tp_count, match_tp_count, current_tp_count, current_ins_count))

            self.before_ins_encoding = after_ins_encoding
            # self.before_unique_ins_ids = after_unique_ins_ids_preds #FATAL BUG!!!!
            self.before_unique_ins_ids = assigned_ids
            self.before_centers = after_centers
            self.before_pred2gt_dict = pred2gt_dict

            matching_list.append({
                'after_xyz': after_xyz[b],
                'after_valid_preds': after_valid_preds[b],
                'after_ins_ids_preds': after_ins_ids_preds[b],
                'after_unique_ins_ids_preds': after_unique_ins_ids_preds,
                'assigned_ids': assigned_ids,
                'prob_matrix': prob_matrix,
                'before_tp_count': before_tp_count,
                'match_tp_count': match_tp_count,
                'current_tp_count': current_tp_count,
                'current_ins_count': current_ins_count,
            })

        return ins_ids_tracking_list, matching_list
