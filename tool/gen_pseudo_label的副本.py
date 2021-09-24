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
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import apex
from tensorboardX import SummaryWriter

from util import dataset, transform, config
from util.reader import DataReader

from tool.gen_pseudo_label_utils import first_forward, second_forward

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)

    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    if args.arch == 'psp':
        assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
    elif args.arch == 'psa':
        if args.compact:
            args.mask_h = (args.train_h - 1) // (8 * args.shrink_factor) + 1
            args.mask_w = (args.train_w - 1) // (8 * args.shrink_factor) + 1
        else:
            assert (args.mask_h is None and args.mask_w is None) or (
                    args.mask_h is not None and args.mask_w is not None)
            if args.mask_h is None and args.mask_w is None:
                args.mask_h = 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1
                args.mask_w = 2 * ((args.train_w - 1) // (8 * args.shrink_factor) + 1) - 1
            else:
                assert (args.mask_h % 2 == 1) and (args.mask_h >= 3) and (
                        args.mask_h <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
                assert (args.mask_w % 2 == 1) and (args.mask_w >= 3) and (
                        args.mask_w <= 2 * ((args.train_h - 1) // (8 * args.shrink_factor) + 1) - 1)
    elif args.arch == 'deeplabv2':
        pass
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def main():
    args = get_parser()
    check(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def local_data_prepare():
    args.data_root = os.path.join(args.local_prefix, args.data_root)
    args.train_labeled_list = os.path.join(args.local_prefix, args.train_labeled_list)
    args.train_unlabeled_list = os.path.join(args.local_prefix, args.train_unlabeled_list)
    args.val_list = os.path.join(args.local_prefix, args.val_list)
    args.test_list = os.path.join(args.local_prefix, args.test_list)
    args.save_pseudo_label_path = os.path.join(args.local_prefix, args.save_pseudo_label_path)
    args.prediction_list = os.path.join(args.local_prefix, args.prediction_list)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    local_data_prepare()

    if args.sync_bn:
        if args.multiprocessing_distributed:
            BatchNorm = apex.parallel.SyncBatchNorm
        else:
            from lib.sync_bn.modules import BatchNorm2d
            BatchNorm = BatchNorm2d
    else:
        BatchNorm = nn.BatchNorm2d
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                                rank=args.rank)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    if args.arch == 'psp':
        from model.pspnet import PSPNet
        model = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, criterion=criterion,
                       BatchNorm=BatchNorm)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.ppm, model.cls, model.aux]
    elif args.arch == 'psa':
        from model.psanet import PSANet
        model = PSANet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, psa_type=args.psa_type,
                       compact=args.compact, shrink_factor=args.shrink_factor, mask_h=args.mask_h, mask_w=args.mask_w,
                       normalization_factor=args.normalization_factor, psa_softmax=args.psa_softmax,
                       criterion=criterion,
                       BatchNorm=BatchNorm)
        modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        modules_new = [model.psa, model.cls, model.aux]
    elif args.arch == 'deeplabv2':
        from model.deeplabv2 import Resnet101_deeplab
        print("args.pretrain data="+args.pretrain_data)
        # import ipdb; ipdb.set_trace(context=20)
        model = Resnet101_deeplab(num_classes=args.classes, criterion=criterion, pretrained=True, pretrain_data=args.pretrain_data)
        modules_ori = model.pretrained_layers()
        modules_new = model.new_layers()

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    args.index_split = 5
    optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)


    global logger, writer
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.unlabelled_batch_size = int(args.unlabelled_batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        if args.use_apex:
            model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level=args.opt_level,
                                                   keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                                   loss_scale=args.loss_scale)
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu])

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_labelled_loader, unlabel_dataReader, gen_pseudo_label_loader, gen_pseudo_label_data, train_labelled_sampler, \
    train_unlabelled_sampler, train_unlabelled_ds = get_labeled_unlabeled_pseudo_dataloader(args)

    epoch = 0
    if args.distributed:
        train_labelled_sampler.set_epoch(epoch)
        train_unlabelled_sampler.set_epoch(epoch)

    # ---------- generate pseudo labels in new folders in args.save_pseudo_label_path
    # ---------- unlabeled list is also generated in the original list folder
    with_module_dict = model.state_dict()
    for key in list(with_module_dict.keys()):
        if 'module.' in key:
            with_module_dict[key.replace('module.', '')] = with_module_dict[key]
            del with_module_dict[key]
    if args.arch == 'psp':
        model_val = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor, pretrained=False)
    elif args.arch == 'deeplabv2':
        model_val = Resnet101_deeplab(num_classes=args.classes, criterion=criterion, pretrained=True,
                                      pretrain_data=args.pretrain_data)
    model_val.load_state_dict(with_module_dict)
    model_val = model_val.cuda()
    pseudo_label_folder = os.path.join(args.save_pseudo_label_path, str(epoch+1))

    name = args.save_pseudo_label_path.split('/')[-2]
    list_path = args.train_unlabeled_list[:-4] + '_' + name + '_logits.txt'

    first_forward(gen_pseudo_label_loader, train_unlabelled_ds.data_list, model_val, args.classes,
                                mean, std, args.base_size, args.test_h, args.test_w, args.scales,
                                pseudo_label_folder, list_path, logger, args)

    args.prediction_list = list_path
    logger.info('prediction list updated as '+args.prediction_list)
    args.save_npy_or_png = 'png'

    list_path = list_path[:-10]+'DARS.txt'
    logger.info('pseudo labels saved in list: ' + list_path)
    # update data loaders
    train_labelled_loader, unlabel_dataReader, gen_pseudo_label_loader, gen_pseudo_label_data, train_labelled_sampler, \
    train_unlabelled_sampler, train_unlabelled_ds = get_labeled_unlabeled_pseudo_dataloader(args)

    second_forward(gen_pseudo_label_loader, train_unlabelled_ds.data_list, model_val, args.classes,
                  mean, std, args.base_size, args.test_h, args.test_w, args.scales,
                  pseudo_label_folder, list_path, logger, args)




def get_labeled_unlabeled_pseudo_dataloader(args):
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        # train_transform = Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])


    gen_pseudo_label_transform = transform.Compose([transform.ToTensor()])

    if args.save_npy_or_png == 'png':
        gen_pseudo_label_data = dataset.SemData_npy(split=args.split, data_root=args.data_root,
                                                    data_list=args.prediction_list,
                                                    # Note: prediction_list npy from now on
                                                    transform=gen_pseudo_label_transform)
    else:
        gen_pseudo_label_data = dataset.SemData(split=args.split, data_root=args.data_root,
                                                data_list=args.train_unlabeled_list,
                                                transform=gen_pseudo_label_transform)

    gen_pseudo_label_loader = torch.utils.data.DataLoader(gen_pseudo_label_data, batch_size=1, shuffle=False,  # naive2+ shuffle here

                                                              num_workers=args.workers,
                                                              pin_memory=True)

    train_labelled_ds = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_labeled_list,
                                        transform=train_transform)


    train_unlabelled_ds = dataset.SemData(split='train', data_root=args.data_root,
                                              data_list=args.train_unlabeled_list,
                                              transform=train_transform)


    print("len(train_labelled_ds)=" + str(len(train_labelled_ds)))
    print("len(train_unlabelled_ds)=" + str(len(train_unlabelled_ds)))

    if args.distributed:
        train_labelled_sampler = torch.utils.data.distributed.DistributedSampler(train_labelled_ds)
        train_unlabelled_sampler = torch.utils.data.distributed.DistributedSampler(train_unlabelled_ds)
    else:
        train_labelled_sampler = None
        train_unlabelled_sampler = None
    train_labelled_loader = torch.utils.data.DataLoader(train_labelled_ds, batch_size=args.batch_size,
                                                        shuffle=(train_labelled_sampler is None),
                                                        num_workers=args.workers, pin_memory=True,
                                                        sampler=train_labelled_sampler,
                                                        drop_last=True)
    train_unlabelled_loader = torch.utils.data.DataLoader(train_unlabelled_ds, batch_size=args.unlabelled_batch_size,
                                                          shuffle=(train_unlabelled_sampler is None),
                                                          num_workers=args.workers, pin_memory=True,
                                                          sampler=train_unlabelled_sampler,
                                                          drop_last=True)
    unlabel_dataReader = DataReader(train_unlabelled_loader)

    return train_labelled_loader, unlabel_dataReader, gen_pseudo_label_loader, gen_pseudo_label_data, train_labelled_sampler, train_unlabelled_sampler, train_unlabelled_ds




if __name__ == '__main__':
    main()

