import os
import random
import time
import cv2, math
import numpy as np
import logging
import argparse
import subprocess


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data

import torch.distributed as dist
import apex

from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, poly_learning_rate_warmup
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def main_process(args):
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def get_small_loss_const(start_iter, current_iter, max_iter, small_loss_const_start, small_loss_const_end):
    const = (small_loss_const_end-small_loss_const_start)/(max_iter-start_iter) * (current_iter-start_iter) + small_loss_const_start
    return const


def train(train_labelled_loader, reader, model, optimizer, epoch, args, writer, logger):
    '''

    :param train_labelled_loader:
    :param reader:
    :param model:
    :param optimizer:
    :param epoch:
    :param args:
    :param writer:
    :param logger:
    :return: all metrics
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()
    super_loss_meter = AverageMeter()  # add
    pseudo_loss_meter = AverageMeter()  # add
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    pse_intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    pse_union_meter = AverageMeter()
    target_meter = AverageMeter()
    valid_ratio_meter = AverageMeter()
    sup_unsup_ratio_meter = AverageMeter()
    sup_extra_ratio_meter = AverageMeter()
    correct_ratio_meter = AverageMeter()  # count the correct percentage of valid area

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_labelled_loader)
    start_iter = args.start_epoch * len(train_labelled_loader)
    reader.construct_iter()

    for i, (input, target) in enumerate(train_labelled_loader):

        data_time.update(time.time() - end)
        current_iter = epoch * len(train_labelled_loader) + i + 1
        if args.zoom_factor != 8:
            h = int((target.size()[1] - 1) / 8 * args.zoom_factor + 1)
            w = int((target.size()[2] - 1) / 8 * args.zoom_factor + 1)
            # 'nearest' mode doesn't support align_corners mode and 'bilinear' mode is fine for downsampling
            target = F.interpolate(target.unsqueeze(1).float(), size=(h, w), mode='bilinear',
                                   align_corners=True).squeeze(1).long()

        input = input.cuda(non_blocking=True)  # input.shape= Bx3xHxW
        target = target.cuda(non_blocking=True)  # TARGET.shape= BxHxW

        unsup_x, unsup_label = reader.read_data()
        unsup_x = unsup_x.cuda(non_blocking=True)
        unsup_label = unsup_label.cuda(non_blocking=True)

        # ------- supervised
        output_pred, main_loss, aux_loss = model(input, target, sup_loss_method=args.sup_loss_method)
        output_pred_x, output_pred_aux = output_pred
        output = output_pred_x.max(1)[1]

        if not args.multiprocessing_distributed:
            main_loss, aux_loss = torch.mean(main_loss), torch.mean(aux_loss)
        supervised_loss = main_loss + args.aux_weight * aux_loss

        # ----------------------------------------------------------------- unsupervised
        if args.only_label is False:
            pse_output_all, pse_main_loss, pse_aux_loss = model(unsup_x, unsup_label,sup_loss_method=args.sup_loss_method) # same as labeled
            pse_output_logits, _ = pse_output_all
            pse_output = pse_output_logits.max(1)[1]

            if not args.multiprocessing_distributed:
                pse_main_loss, pse_aux_loss = torch.mean(pse_main_loss), torch.mean(pse_aux_loss)
            pseudo_loss = pse_main_loss + args.aux_weight * pse_aux_loss

            # get pse mIou between pseudo_label and gt
            intersection_per, union_per, _ = intersectionAndUnionGPU(pse_output, unsup_label,
                                                                     args.classes, args.ignore_label)
            if args.multiprocessing_distributed:
                dist.all_reduce(intersection_per), dist.all_reduce(union_per)
            intersection_per, union_per = intersection_per.detach().cpu().numpy(), union_per.detach().cpu().numpy()
            pse_intersection_meter.update(intersection_per), pse_union_meter.update(union_per)

        elif args.only_label:
            pseudo_loss= torch.zeros(1).cuda()

        loss = supervised_loss + args.consistency_lambda * pseudo_loss

        optimizer.zero_grad()
        if args.use_apex and args.multiprocessing_distributed:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        n = input.size(0)
        if args.multiprocessing_distributed:
            main_loss, aux_loss, loss = main_loss.detach() * n, aux_loss * n, loss * n  # not considering ignore pixels
            pseudo_loss = pseudo_loss * n * args.unlabelled_batch_size / args.batch_size
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(main_loss), dist.all_reduce(aux_loss), dist.all_reduce(loss), dist.all_reduce(
                count)
            dist.all_reduce(pseudo_loss)
            n = count.item()
            main_loss, aux_loss, loss = main_loss / n, aux_loss / n, loss / n
            pseudo_loss = pseudo_loss / (n * args.unlabelled_batch_size / args.batch_size)

        # get mIou for supervised part
        intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        main_loss_meter.update(main_loss.item(), n)
        aux_loss_meter.update(aux_loss.item(), n)
        super_loss_meter.update(supervised_loss.item(), n)
        pseudo_loss_meter.update(pseudo_loss.item(), n * args.unlabelled_batch_size / args.batch_size)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)

        # logger.info('current lr = {}'.format(current_lr))
        for index in range(0, args.index_split):  # index_split=5
            optimizer.param_groups[index]['lr'] = current_lr
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process(args) :
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Sup-Loss {super_loss_meter.avg:.4f} '
                        'Pseudo-loss {pseudo_loss_meter.avg:.4f} '
                        'pse mIou {pse_mIou:.4f} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_labelled_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          main_loss_meter=main_loss_meter,
                                                          aux_loss_meter=aux_loss_meter,
                                                          super_loss_meter=super_loss_meter,
                                                          pseudo_loss_meter=pseudo_loss_meter,
                                                          pse_mIou=np.mean(pse_intersection_meter.sum / (
                                                                  pse_union_meter.sum + 1e-10)),
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process(args) and writer is not None:
            writer.add_scalar('total_loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('supervised_loss_train_batch', super_loss_meter.val, current_iter)
            writer.add_scalar('pseudo_loss_train_batch', pseudo_loss_meter.val, current_iter)
            writer.add_scalar('sup_unsup_ratio_train_batch', sup_unsup_ratio_meter.val, current_iter)
            writer.add_scalar('pse_mIoU_train_batch',
                            np.mean(pse_intersection_meter.sum / (pse_union_meter.sum + 1e-10)), current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(intersection / (target + 1e-10)), current_iter)
            # writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process(args):
        logger.info(
            'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc/suploss/pseloss/pse_mIoU {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
                epoch + 1, args.epochs, mIoU,
                mAcc, allAcc, super_loss_meter.avg, pseudo_loss_meter.avg, np.mean(pse_intersection_meter.sum / (pse_union_meter.sum + 1e-10))))
    return main_loss_meter.avg, loss_meter.avg, super_loss_meter.avg, pseudo_loss_meter.avg, mIoU, mAcc, allAcc, valid_ratio_meter.avg, correct_ratio_meter.avg, np.mean(
        pse_intersection_meter.sum / (pse_union_meter.sum + 1e-10)), sup_unsup_ratio_meter.avg




