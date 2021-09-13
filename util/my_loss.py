import os
import random
import time
import cv2, math
import numpy as np
import logging
import argparse
import subprocess


import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Subset, Dataset
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from tensorboardX import SummaryWriter

import torchvision
from torchvision.transforms import Compose
from util import dataset, transform, config, augmentation, my_transform
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, poly_learning_rate_warmup
from util import ramps
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_kl_dis_of_same_shape(logits1,logits2):
    '''

    :param logits1: BCHW
    :param logits2: BCHW
    :return: BHW
    '''

    probas1 = F.log_softmax(logits1, dim=1)
    probas2 = F.softmax(logits2, dim=1)

    same_shape_as_input = F.kl_div(probas1, probas2, reduction='none')
    kl_dis_matrix = torch.sum(same_shape_as_input, dim=1)
    return kl_dis_matrix


def get_smoonth_loss(logits, img):
    '''
    edge aware smoonth loss for unlabeled data
    :param logits: BCHW
    :param img: B3HW
    :return: (1)
    '''
    # import ipdb
    # ipdb.set_trace(context=20)
    kl_x = get_kl_dis_of_same_shape(logits[:,:,:,:-1], logits[:,:,:,1:])
    kl_y = get_kl_dis_of_same_shape(logits[:,:,:-1,:], logits[:,:,1:,:])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_img_x = torch.squeeze(grad_img_x)
    grad_img_y = torch.squeeze(grad_img_y)

    kl_x *= torch.exp(-grad_img_x)
    kl_y *= torch.exp(-grad_img_y)

    return kl_x.mean() + kl_y.mean()


def get_soft_label_weight_dict_from_npy(max_cf_in_soft_label=0.8):
    # every col in soft_matrix is the soft label value to replace hard label
    # npy_path = args.confusion_matrix_path
    # npy_path = 'initmodel/372_labeled_confusion_matrix_argmax.npy'  # setting 0.63
    npy_path = 'initmodel/confusion_matrix_372_0.60_argmax.npy'  # setting 0.60
    confusion_matrix = np.load(npy_path)
    for i in range(19):
        confusion_matrix[i, i] = 0

    normalized_confusion_matrix = confusion_matrix / confusion_matrix.sum(0)
    soft_matrix = np.identity(19) * max_cf_in_soft_label

    # if only_hard_class:
    #     easy_class = [0,1,2,8,10,13]
    #     for j in easy_class:
    #         normalized_confusion_matrix[:,j] = 0
    #         normalized_confusion_matrix[j,j] = 1
    soft_matrix += normalized_confusion_matrix * (1 - max_cf_in_soft_label)
    # import ipdb
    # ipdb.set_trace(context=20)
    return soft_matrix


def hard_label_to_soft_label(hard_label, max_cf_in_soft_label=0.8):
    '''

    :param hard_label: B*H*W
    :param max_cf_in_soft_label: max value in soft label
    :return: soft_label B*H*W*19
    '''
    soft_matrix = get_soft_label_weight_dict_from_npy(max_cf_in_soft_label)
    soft_label = torch.ones((hard_label.shape[0], hard_label.shape[1], hard_label.shape[2], 19))*255
    soft_label = soft_label.cuda()

    # import ipdb
    # ipdb.set_trace(context=20)
    for j in range(19):
        class_j_position = (hard_label == j)
        soft_label[class_j_position] = torch.from_numpy(soft_matrix[:, j]).float().cuda()

    return soft_label


def hard_label_to_soft_label_top2(hard_label, second_max, max_cf_in_soft_label=0.8):
    '''

    :param hard_label: B*H*W
    :param second_max: B*H*W
    :param max_cf_in_soft_label: max value in soft label
    :return: soft_label B*H*W*19
    '''
    soft_label = torch.ones((hard_label.shape[0], hard_label.shape[1], hard_label.shape[2], 19)) * 255
    soft_label = soft_label.cuda()

    high_cf_mask = (hard_label == 255)
    second_max[high_cf_mask] = 255

    for j in range(19):

        class_j_position_max = (hard_label == j)
        vec_max = torch.zeros(19).cuda()
        vec_max[j] = max_cf_in_soft_label
        soft_label[class_j_position_max] = vec_max

    for j in range(19):

        class_j_position_second_max = (second_max == j)
        vec_second_max = torch.zeros(19).cuda()
        vec_second_max[j] = 1 - max_cf_in_soft_label
        soft_label[class_j_position_second_max] += vec_second_max

    # import ipdb
    # ipdb.set_trace(context=20)
    return soft_label



