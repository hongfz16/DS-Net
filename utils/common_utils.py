import numpy as np
import torch
import random
import logging
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import subprocess
import pickle
import shutil
from scipy import stats as s
import numba as nb
from .evaluate_panoptic import class_inv_lut

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick: 0 - 1 = 255

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def grp_range_torch(a,dev):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)
    # generate array like [0,1,2,3,4,5,0,1,2,3,4,5,6] where each 0-n gives id to points inside the same grid

def parallel_FPS(np_cat_fea,K):
    return  nb_greedy_FPS(np_cat_fea,K)

# @nb.jit('b1[:](f4[:,:],i4)',nopython=True,cache=True)
def nb_greedy_FPS(xyz,K):
    start_element = 0
    sample_num = xyz.shape[0]
    sum_vec = np.zeros((sample_num,1),dtype = np.float32)
    xyz_sq = xyz**2
    for j in range(sample_num):
        sum_vec[j,0] = np.sum(xyz_sq[j,:])
    pairwise_distance = sum_vec + np.transpose(sum_vec) - 2*np.dot(xyz, np.transpose(xyz))

    candidates_ind = np.zeros((sample_num,),dtype = np.bool_)
    candidates_ind[start_element] = True
    remain_ind = np.ones((sample_num,),dtype = np.bool_)
    remain_ind[start_element] = False
    all_ind = np.arange(sample_num)

    for i in range(1,K):
        if i == 1:
            min_remain_pt_dis = pairwise_distance[:,start_element]
            min_remain_pt_dis = min_remain_pt_dis[remain_ind]
        else:
            cur_dis = pairwise_distance[remain_ind,:]
            cur_dis = cur_dis[:,candidates_ind]
            min_remain_pt_dis = np.zeros((cur_dis.shape[0],),dtype = np.float32)
            for j in range(cur_dis.shape[0]):
                min_remain_pt_dis[j] = np.min(cur_dis[j,:])
        next_ind_in_remain = np.argmax(min_remain_pt_dis)
        next_ind = all_ind[remain_ind][next_ind_in_remain]
        candidates_ind[next_ind] = True
        remain_ind[next_ind] = False

    return candidates_ind

def merge_ins_sem(_sem, ins, ins_classified_labels=None, ins_classified_ids=None, merge_pred_unlabeled=True, merge_few_pts_ins=True):
    sem = _sem.copy()
    ins_ids = np.unique(ins)
    for id in ins_ids:
        if id == 0: # id==0 means stuff classes
            continue
        ind = (ins == id)
        if not merge_few_pts_ins:
            if np.sum(ind) < 50:
                continue
        if ins_classified_labels is None:
            sub_sem = sem[ind]
            mode_sem_id = int(s.mode(sub_sem)[0])
            sem[ind] = mode_sem_id
        else:
            if id in ins_classified_ids:
                curr_classified_id = ins_classified_labels[ins_classified_ids==id][0]
                # mode_sem_id = ins_classified_labels[ins_classified_ids==id][0]
                if not merge_pred_unlabeled and curr_classified_id == 0: #TODO: change to use cfg
                    continue
                mode_sem_id = curr_classified_id
                sem[ind] = mode_sem_id
            else:
                sub_sem = sem[ind]
                mode_sem_id = int(s.mode(sub_sem)[0])
                sem[ind] = mode_sem_id
    return sem

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def init_dist_slurm(batch_size, tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        batch_size:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    assert batch_size % total_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, total_gpus)
    batch_size_each_gpu = batch_size // total_gpus
    rank = dist.get_rank()
    return batch_size_each_gpu, rank


def init_dist_pytorch(batch_size, tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    assert batch_size % num_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, num_gpus)
    batch_size_each_gpu = batch_size // num_gpus
    rank = dist.get_rank()
    return batch_size_each_gpu, rank

def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def merge_evaluator(evaluator, tmp_dir, prefix=''):
    rank, world_size = get_dist_info()
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    dist.barrier()
    pickle.dump(evaluator, open(os.path.join(tmp_dir, '{}evaluator_part_{}.pkl'.format(prefix, rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    for i in range(1, world_size):
        part_file = os.path.join(tmp_dir, '{}evaluator_part_{}.pkl'.format(prefix, i))
        evaluator.merge(pickle.load(open(part_file, 'rb')))

    return evaluator

def save_test_results(ret_dict, output_dir, batch):
    assert len(ret_dict['sem_preds']) == 1

    sem_preds = ret_dict['sem_preds'][0]
    ins_preds = ret_dict['ins_preds'][0]

    sem_inv = class_inv_lut[sem_preds].astype(np.uint32)
    label = sem_inv.reshape(-1, 1) + ((ins_preds.astype(np.uint32) << 16) & 0xFFFF0000).reshape(-1, 1)

    pcd_path = batch['pcd_fname'][0]
    seq = pcd_path.split('/')[-3]
    pcd_fname = pcd_path.split('/')[-1].split('.')[-2]+'.label'
    fname = os.path.join(output_dir, seq, 'predictions', pcd_fname)
    label.reshape(-1).astype(np.uint32).tofile(fname)

def safe_vis(xyz, labels, ignore_zero=True):
    pass
