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

from util import dataset, transform, config, augmentation, reader
from util.reader import DataReader
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from tool.task3.train_func import train
from util.validate_full_size import test, cal_acc
from util.util import AverageMeter, intersectionAndUnion, check_makedirs, colorize

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


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
        port = find_free_port()
        args.dist_url = f"tcp://127.0.0.1:{port}"
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
    args.prediction_list = os.path.join(args.local_prefix, args.prediction_list)
    cmd_line = "mkdir -p {0}".format(args.save_folder + '/gray')
    subprocess.call(cmd_line.split())


def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss

    args.save_path = args.save_path + args.exp_name + '/model'
    args.save_folder = args.save_folder + args.exp_name + '/result/val'
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
        print("args.pretrain data=" + args.pretrain_data)
        # import ipdb; ipdb.set_trace(context=20)
        model = Resnet101_deeplab(num_classes=args.classes, criterion=criterion, pretrained=True,
                                  pretrain_data=args.pretrain_data)
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

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_labelled_loader, unlabel_dataReader, train_labelled_sampler, \
    train_unlabelled_sampler = get_labeled_unlabeled_pseudo_dataloader(args)

    if args.evaluate:
        if args.evaluate_full_size is False:
            val_transform = transform.Compose([
                transform.Crop([args.train_h, args.train_w], crop_type='center', padding=mean,
                               ignore_label=args.ignore_label),
                transform.ToTensor(),
                transform.Normalize(mean=mean, std=std)])
            if args.data_root == 'gta5':
                val_data = dataset.SemData(split='val', data_root='cityscapes', data_list=args.val_list,
                                           transform=val_transform)
            else:
                val_data = dataset.SemData(split='val', data_root=args.data_root, data_list=args.val_list,
                                           transform=val_transform)
            if args.distributed:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                     num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        else:
            gray_folder = os.path.join(args.save_folder, 'gray')
            color_folder = os.path.join(args.save_folder, 'color')
            val_transform = transform.Compose([transform.ToTensor()])
            val_data = dataset.SemData(split=args.split, data_root=args.data_root, data_list=args.val_list,
                                       transform=val_transform)

            val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=args.workers,
                                                     pin_memory=True)
            colors = np.loadtxt(args.colors_path).astype('uint8')
            names = [line.rstrip('\n') for line in open(args.names_path)]

            if args.arch == 'psp':
                model_val = PSPNet(layers=args.layers, classes=args.classes, zoom_factor=args.zoom_factor,
                                   pretrained=False)
            elif args.arch == 'deeplabv2':
                model_val = Resnet101_deeplab(num_classes=args.classes, criterion=criterion, pretrained=True,
                                              pretrain_data=args.pretrain_data)

    # ----- init save best ckpt vars
    best_val_miou = args.evaluate_previous_best_val_mIou
    best_ckpt_name = None

    # ---- for the file exist issue at slurm
    check_makedirs(gray_folder)
    # ---- for the file exist issue at slurm

    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        if args.distributed:
            train_labelled_sampler.set_epoch(epoch)
            train_unlabelled_sampler.set_epoch(epoch)

        try:
            writer
        except NameError:
            writer = None

        try:
            list_path
        except NameError:
            list_path = None

        main_loss_train, total_loss_train, supervised_loss_train, consistency_loss_train, mIoU_train, mAcc_train, \
        allAcc_train, valid_ratio, correct_ratio, pse_miou, loss_ratio = \
            train(train_labelled_loader, unlabel_dataReader, model, optimizer, epoch, args, writer, logger)

        if main_process():
            # writer.add_scalar('main_loss_train', main_loss_train, epoch_log)
            writer.add_scalar('total_loss_train', total_loss_train, epoch_log)
            writer.add_scalar('supervised_loss_train', supervised_loss_train, epoch_log)
            writer.add_scalar('pseudo_loss_train', consistency_loss_train, epoch_log)
            writer.add_scalar('valid_ratio_train', valid_ratio, epoch_log)
            writer.add_scalar('correct_ratio_train', correct_ratio, epoch_log)
            writer.add_scalar('pse_mIoU_train', pse_miou, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('loss_ratio_train', loss_ratio, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       filename)

        if args.evaluate and (epoch_log > args.evaluate_start) and (
                (epoch_log - args.evaluate_start) % args.evaluate_freq == 0) and main_process():
            if args.evaluate_full_size is False:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(val_loader, model, criterion)
            else:
                with_module_dict = model.state_dict()
                for key in list(with_module_dict.keys()):
                    if 'module.' in key:
                        with_module_dict[key.replace('module.', '')] = with_module_dict[key]
                        del with_module_dict[key]
                model_val.load_state_dict(with_module_dict)
                model_val = model_val.cuda()
                gray_folder_ = os.path.join(gray_folder, str(epoch_log))
                mIoU_val, mAcc_val, allAcc_val = test(val_loader, val_data.data_list, model_val, args.classes, mean,
                                                      std,
                                                      args.base_size, args.test_h,
                                                      args.test_w, args.scales, gray_folder_, color_folder, colors,
                                                      names, args)
                loss_val = 0
            # -------- save best val mIou ckpt
            if mIoU_val > best_val_miou and main_process():
                if best_ckpt_name is not None:
                    if os.path.exists(best_ckpt_name):
                        # os.remove(best_ckpt_name)
                        logger.info('Remove checkpoint: ' + best_ckpt_name)
                best_val_miou = mIoU_val
                filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
                if not os.path.exists(filename):
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename)
                best_ckpt_name = filename

            elif mIoU_val <= best_val_miou and main_process():
                filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
                if not os.path.exists(filename):
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)

    logger.info("finish " + args.save_path)


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

    unlabeled_train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    train_labelled_ds = dataset.SemData(split='train', data_root=args.data_root, data_list=args.train_labeled_list,
                                        transform=train_transform)

    train_unlabelled_ds = dataset.SemData(split='train', data_root=args.data_root,
                                          data_list=args.train_unlabeled_list,
                                          transform=unlabeled_train_transform)

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

    return train_labelled_loader, unlabel_dataReader, train_labelled_sampler, train_unlabelled_sampler


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        output = model(input)
        if args.zoom_factor != 8:
            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
        loss = criterion(output, target)

        n = input.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)

        output = output.max(1)[1]
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    main()

