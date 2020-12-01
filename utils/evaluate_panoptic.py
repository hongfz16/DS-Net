import argparse
import os
import yaml
import sys
import numpy as np
import time
import json
from .eval_np import PanopticEval
from .config import global_cfg
need_nuscenes_remap = False

if global_cfg.DATA_CONFIG.DATASET_NAME == 'SemanticKitti':
    DATA = yaml.safe_load(open('semantic-kitti.yaml', 'r'))
    # get number of interest classes, and the label mappings
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)
    class_strings = DATA["labels"]
    # make lookup table for mapping
    maxkey = max(class_remap.keys())
    # +100 hack making lut bigger just in case there are unknown labels
    class_lut = np.zeros((maxkey + 100), dtype=np.int32)
    class_lut[list(class_remap.keys())] = list(class_remap.values())
    ignore_class = [cl for cl, ignored in class_ignore.items() if ignored]

    class_inv_lut = np.zeros((20), dtype=np.int32)
    class_inv_lut[list(class_inv_remap.keys())] = list(class_inv_remap.values())

    things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
    stuff = [
        'road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole',
        'traffic-sign'
    ]
    all_classes = things + stuff
    valid_xentropy_ids = [1, 4, 2, 3, 5, 6, 7, 8]
else:
    raise NotImplementedError

def init_eval(min_points = 50):
    print("New evaluator with min_points of {}".format(min_points))
    class_evaluator = PanopticEval(nr_classes, None, ignore_class, min_points = min_points)
    return class_evaluator

def eval_one_scan(class_evaluator, gt_sem, gt_ins, pred_sem, pred_ins):
    class_evaluator.addBatch(pred_sem, pred_ins, gt_sem, gt_ins)

def eval_one_scan_w_fname(class_evaluator, gt_sem, gt_ins, pred_sem, pred_ins, fname):
    class_evaluator.addBatch_w_fname(pred_sem, pred_ins, gt_sem, gt_ins, fname)

def printResults(class_evaluator, logger=None, sem_only=False):
    class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = class_evaluator.getPQ()
    class_IoU, class_all_IoU = class_evaluator.getSemIoU()

    # now make a nice dictionary
    output_dict = {}

    # make python variables
    class_PQ = class_PQ.item()
    class_SQ = class_SQ.item()
    class_RQ = class_RQ.item()
    class_all_PQ = class_all_PQ.flatten().tolist()
    class_all_SQ = class_all_SQ.flatten().tolist()
    class_all_RQ = class_all_RQ.flatten().tolist()
    class_IoU = class_IoU.item()
    class_all_IoU = class_all_IoU.flatten().tolist()

    output_dict["all"] = {}
    output_dict["all"]["PQ"] = class_PQ
    output_dict["all"]["SQ"] = class_SQ
    output_dict["all"]["RQ"] = class_RQ
    output_dict["all"]["IoU"] = class_IoU

    classwise_tables = {}

    for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_PQ, class_all_RQ, class_all_SQ, class_all_IoU)):
        class_str = class_strings[class_inv_remap[idx]]
        output_dict[class_str] = {}
        output_dict[class_str]["PQ"] = pq
        output_dict[class_str]["SQ"] = sq
        output_dict[class_str]["RQ"] = rq
        output_dict[class_str]["IoU"] = iou

    PQ_all = np.mean([float(output_dict[c]["PQ"]) for c in all_classes])
    PQ_dagger = np.mean([float(output_dict[c]["PQ"]) for c in things] + [float(output_dict[c]["IoU"]) for c in stuff])
    RQ_all = np.mean([float(output_dict[c]["RQ"]) for c in all_classes])
    SQ_all = np.mean([float(output_dict[c]["SQ"]) for c in all_classes])

    PQ_things = np.mean([float(output_dict[c]["PQ"]) for c in things])
    RQ_things = np.mean([float(output_dict[c]["RQ"]) for c in things])
    SQ_things = np.mean([float(output_dict[c]["SQ"]) for c in things])

    PQ_stuff = np.mean([float(output_dict[c]["PQ"]) for c in stuff])
    RQ_stuff = np.mean([float(output_dict[c]["RQ"]) for c in stuff])
    SQ_stuff = np.mean([float(output_dict[c]["SQ"]) for c in stuff])
    mIoU = output_dict["all"]["IoU"]

    codalab_output = {}
    codalab_output["pq_mean"] = float(PQ_all)
    codalab_output["pq_dagger"] = float(PQ_dagger)
    codalab_output["sq_mean"] = float(SQ_all)
    codalab_output["rq_mean"] = float(RQ_all)
    codalab_output["iou_mean"] = float(mIoU)
    codalab_output["pq_stuff"] = float(PQ_stuff)
    codalab_output["rq_stuff"] = float(RQ_stuff)
    codalab_output["sq_stuff"] = float(SQ_stuff)
    codalab_output["pq_things"] = float(PQ_things)
    codalab_output["rq_things"] = float(RQ_things)
    codalab_output["sq_things"] = float(SQ_things)

    key_list = [
        "pq_mean",
        "pq_dagger",
        "sq_mean",
        "rq_mean",
        "iou_mean",
        "pq_stuff",
        "rq_stuff",
        "sq_stuff",
        "pq_things",
        "rq_things",
        "sq_things"
    ]

    if sem_only and logger != None:
        evaluated_fnames = class_evaluator.evaluated_fnames
        logger.info('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        logger.info('|        |  IoU   |   PQ   |   RQ   |   SQ   |')
        for k, v in output_dict.items():
            logger.info('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k.ljust(8)[-8:], v['IoU'], v['PQ'], v['RQ'], v['SQ']
            ))
        return codalab_output
    if sem_only and logger is None:
        evaluated_fnames = class_evaluator.evaluated_fnames
        print('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        print('|        |  IoU   |   PQ   |   RQ   |   SQ   |')
        for k, v in output_dict.items():
            print('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k.ljust(8)[-8:], v['IoU'], v['PQ'], v['RQ'], v['SQ']
            ))
        return codalab_output

    if logger != None:
        evaluated_fnames = class_evaluator.evaluated_fnames
        logger.info('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        logger.info('|        |   PQ   |   RQ   |   SQ   |  IoU   |')
        for k, v in output_dict.items():
            logger.info('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k.ljust(8)[-8:], v['PQ'], v['RQ'], v['SQ'], v['IoU']
            ))
        logger.info('True Positive: ')
        logger.info('\t|\t'.join([str(x) for x in class_evaluator.pan_tp]))
        logger.info('False Positive: ')
        logger.info('\t|\t'.join([str(x) for x in class_evaluator.pan_fp]))
        logger.info('False Negative: ')
        logger.info('\t|\t'.join([str(x) for x in class_evaluator.pan_fn]))
    if logger is None:
        evaluated_fnames = class_evaluator.evaluated_fnames
        print('Evaluated {} frames. Duplicated frame number: {}'.format(len(evaluated_fnames), len(evaluated_fnames) - len(set(evaluated_fnames))))
        print('|        |   PQ   |   RQ   |   SQ   |  IoU   |')
        for k, v in output_dict.items():
            print('|{}| {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                k.ljust(8)[-8:], v['PQ'], v['RQ'], v['SQ'], v['IoU']
            ))
        print('True Positive: ')
        print('\t|\t'.join([str(x) for x in class_evaluator.pan_tp]))
        print('False Positive: ')
        print('\t|\t'.join([str(x) for x in class_evaluator.pan_fp]))
        print('False Negative: ')
        print('\t|\t'.join([str(x) for x in class_evaluator.pan_fn]))

    for key in key_list:
        if logger != None:
            logger.info("{}:\t{}".format(key, codalab_output[key]))
        else:
            print("{}:\t{}".format(key, codalab_output[key]))

    return codalab_output
