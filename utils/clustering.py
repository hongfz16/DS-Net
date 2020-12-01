import numpy as np
import hdbscan
from sklearn.cluster import MeanShift, DBSCAN
from functools import partial
import torch

def hdbscan_cluster(shifted_pcd, valid, min_cluster_size=50):
    clustered_ins_ids = np.zeros(shifted_pcd.shape[0], dtype=np.int32)
    valid_shifts = shifted_pcd[valid, :].reshape(-1, 3)
    if valid_shifts.shape[0] == 0:
        return clustered_ins_ids
    cluster = hdbscan.HDBSCAN(
        min_cluster_size = min_cluster_size,
        allow_single_cluster = True
    ).fit(valid_shifts)
    instance_labels = cluster.labels_
    instance_labels += (-instance_labels.min() + 1)
    clustered_ins_ids[valid] = instance_labels
    return clustered_ins_ids

def dbscan_cluster(shifted_pcd, valid, min_samples=25, eps=0.2):
    embedding_dim = shifted_pcd.shape[1]
    clustered_ins_ids = np.zeros(shifted_pcd.shape[0], dtype=np.int32)
    valid_shifts = shifted_pcd[valid, :].reshape(-1, embedding_dim) if valid is not None else shifted_pcd
    if valid_shifts.shape[0] == 0:
        return clustered_ins_ids

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(valid_shifts)
    labels = dbscan.labels_ - dbscan.labels_.min() + 1
    assert np.min(labels) > 0
    if valid is not None:
        clustered_ins_ids[valid] = labels
        return clustered_ins_ids
    else:
        return labels

def meanshift_cluster(shifted_pcd, valid, bandwidth=1.0):
    embedding_dim = shifted_pcd.shape[1]
    clustered_ins_ids = np.zeros(shifted_pcd.shape[0], dtype=np.int32)
    valid_shifts = shifted_pcd[valid, :].reshape(-1, embedding_dim) if valid is not None else shifted_pcd
    if valid_shifts.shape[0] == 0:
        return clustered_ins_ids

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    try:
        ms.fit(valid_shifts)
    except Exception as e:
        ms = MeanShift(bandwidth=bandwidth)
        ms.fit(valid_shifts)
        print("\nException: {}.".format(e))
        print("Disable bin_seeding.")
    labels = ms.labels_ + 1
    assert np.min(labels) > 0
    if valid is not None:
        clustered_ins_ids[valid] = labels
        return clustered_ins_ids
    else:
        return labels

def pairwise_distance(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).reshape(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).reshape(1, -1)
    else:
        y = x
        y_norm = x_norm.reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * np.matmul(x, y.T)
    return dist

def cluster_batch(cart_xyz_list, shift_list, valid_list, params, choose_algo='meanshift', other=None):
    bs = len(cart_xyz_list)
    pred_ins_ids_list = []
    for i in range(bs):
        if choose_algo == 'meanshift':
            bandwidth = params['bandwidth']
            i_clustered_ins_ids = meanshift_cluster(cart_xyz_list[i] + shift_list[i], valid_list[i], bandwidth)
        elif choose_algo == 'meanshift_embedding':
            bandwidth = params['bandwidth']
            i_clustered_ins_ids = meanshift_cluster(shift_list[i], valid_list[i], bandwidth)
        elif choose_algo == 'hdbscan':
            min_cluster_size = params['min_cluster_size']
            i_clustered_ins_ids = hdbscan_cluster(cart_xyz_list[i] + shift_list[i], valid_list[i], min_cluster_size)
        elif choose_algo == 'dbscan':
            min_samples = params['min_samples']
            eps = params['eps']
            i_clustered_ins_ids = dbscan_cluster(cart_xyz_list[i] + shift_list[i], valid_list[i], min_samples=min_samples, eps=eps)
        else:
            raise NotImplementedError
        pred_ins_ids_list.append(i_clustered_ins_ids)
    return pred_ins_ids_list

def MeanShift_cluster(cfg):
    bandwidth = cfg.MODEL.POST_PROCESSING.BANDWIDTH
    return partial(cluster_batch, choose_algo='meanshift', params={'bandwidth': bandwidth})

def HDBSCAN_cluster(cfg):
    min_cluster_size = cfg.MODEL.POST_PROCESSING.MIN_CLUSTER_SIZE
    return partial(cluster_batch, choose_algo='hdbscan', params={'min_cluster_size': min_cluster_size})

def DBSCAN_cluster(cfg):
    min_samples = cfg.MODEL.POST_PROCESSING.MIN_SAMPLES
    eps = cfg.MODEL.POST_PROCESSING.EPS
    return partial(cluster_batch, choose_algo='dbscan', params={'min_samples': min_samples, 'eps': eps})

def MeanShift_embedding_cluster(cfg):
    bandwidth = cfg.MODEL.POST_PROCESSING.BANDWIDTH
    return partial(cluster_batch, choose_algo='meanshift_embedding', params={'bandwidth': bandwidth})
