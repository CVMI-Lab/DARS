import os
import time
import logging
import argparse
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from util import dataset, transform, config
from util.util import AverageMeter, check_makedirs

cv2.ocl.setUseOpenCL(False)



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


def check(args):
    assert args.classes > 1
    assert args.zoom_factor in [1, 2, 4, 8]
    assert args.split in ['train', 'val', 'test']
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
    else:
        raise Exception('architecture not supported yet'.format(args.arch))


def net_process(model, image, mean, std=None, flip=True):
    input = torch.from_numpy(image.transpose((2, 0, 1))).float()
    if std is None:
        for t, m in zip(input, mean):
            t.sub_(m)
    else:
        for t, m, s in zip(input, mean, std):
            t.sub_(m).div_(s)
    input = input.unsqueeze(0).cuda()
    if flip:
        input = torch.cat([input, input.flip(3)], 0)
    with torch.no_grad():
        output = model(input)
    _, _, h_i, w_i = input.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear', align_corners=True)
    # output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean, std=None, stride_rate=2 / 3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            prediction_crop[s_h:e_h, s_w:e_w, :] += net_process(model, image_crop, mean, std)
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return prediction


def main_process(args):
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def get_threshold_and_sampling_ratio(args, logger):
    global_num = args.global_num
    dataset = args.pseudo_data
    split = args.semi_split
    describ = args.npy_describ
    class_num = args.classes
    npy_path_pre = 'exp/npy_files/' + describ
    labeling_ratio = args.labeling_ratio
    thres = []
    sp = []
    global_cf_dict = [np.array([], dtype=np.float16) for _ in range(class_num)]
    logger.info('Sorting ...'+dataset+' split '+split+' labeling ratio='+str(labeling_ratio))

    mode = args.thresholds_method

    if dataset == 'cityscapes' and split == '8':
        all_pixel_num = 1024 * 2048
        valid_ratio = 88.04
        class_ratio_prior_list = [32.07, 5.71, 20.7, 0.564, 0.761, 1.054, 0.1696, 0.5014, 13.4993, 0.8981, 3.6445,
                                  1.1458,0.1393, 6.0, 0.2949, 0.1954, 0.2341, 0.0818, 0.3917]
    elif dataset == 'cityscapes' and split == '4':
        all_pixel_num = 1024 * 2048
        valid_ratio = 88.14
        class_ratio_prior_list = [32.4802214740425, 5.500137357301609, 20.27850381789669, 0.5930969151117469,
                                  0.6850259278410225,
                                  1.0662038480081866, 0.17537673314412433, 0.5036413028676022, 13.991073831435171,
                                  1.0374317887008833,
                                  3.543039162953695, 1.1905784888934063, 0.12658552456927555, 5.9728318645108125,
                                  0.21484167345108526,
                                  0.18578331957581223, 0.14823566200912638, 0.08569032915176883, 0.36252775499897616]
    elif dataset == 'voc2012' and split == '1.4k':
        all_pixel_num = 500 * 375
        valid_ratio = 95.55735187541127
        class_ratio_prior_list = [65.73545514367186, 0.9092299773342106, 0.8707718944212912, 0.8492143013818819,
                                  0.6514714630401404,
                                  0.485292330189369, 1.0943523872194196, 1.9926919646121224, 3.19925597718798,
                                  1.2808680266140235,
                                  0.4920756306207502, 0.9676409738977846, 2.952389208159684, 0.8709638078525992,
                                  1.1353629597133874,
                                  7.8584497769978805, 0.5948343057688089, 0.5438430649996344, 1.1204710389705346,
                                  1.2274377714411056, 0.7252798713168094]
    elif dataset == 'voc2012' and split == '8':
        all_pixel_num = 500 * 375
        valid_ratio = 94.79185839636911
        class_ratio_prior_list = [65.5032008068583, 0.9112766515380736, 0.8744524457892083, 0.8823354513363589,
                                  0.5325793242561775, 0.5341938477054967,
                                  1.2314271306101867, 2.088203731719617, 2.918815128593041, 0.9957426122037316,
                                  0.42744246091780136, 0.8034674735249623,
                                  2.7000649520927884, 0.7574846192637418, 1.1803102370146243, 8.489111850731215,
                                  0.5730634392334847, 0.5158753403933435,
                                  1.1186291477559254, 0.9201803328290469, 0.834001412002017]

    if mode == 'DARS':
        for j in range(class_num):
            class_j_num = int(float(labeling_ratio) * float(class_ratio_prior_list[j]) / valid_ratio * all_pixel_num * int(global_num))
            npy_path = npy_path_pre+'/class_'+ str(j)+'.npy'
            global_cf_dict[j] = np.load(npy_path)
            global_cf_dict[j].sort()
            global_cf_dict[j] = global_cf_dict[j][::-1]  # check if big to small

            if class_j_num > len(global_cf_dict[j]):
                thres.append(global_cf_dict[j][len(global_cf_dict[j]) - 1].item())
                print('class'+str(j)+'overflow')
            else:
                thres.append(global_cf_dict[j][class_j_num - 1].item())

            thres_j = thres[j]
            remain = (global_cf_dict[j] >= thres_j)
            sp_j = float(class_j_num) / float(remain.sum())
            if sp_j >1:
                sp_j = 1
            sp.append(sp_j)

            logger.info(thres)

        logger.info('sp:')
        logger.info(sp)

    elif mode == 'cbst':
        for j in range(class_num):
            npy_path = npy_path_pre+'/class_'+ str(j)+'.npy'
            global_cf_dict[j] = np.load(npy_path)
            global_cf_dict[j].sort()
            thres.append(global_cf_dict[j][-int(len(global_cf_dict[j]) * float(labeling_ratio)) - 1].item())
            logger.info(thres)
            sp = [1.0 for i in range(class_num)]
    return thres, sp

# first forward
def first_forward(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales,
                                gray_folder, list_path, logger, args):
    if main_process(args):
        logger.info('>>>>>>>>>>>>>>>> Generate pseudo labels >>>>>>>>>>>>>>>>')
    list_writer = open(list_path, 'a')
    batch_time = AverageMeter()
    threshold_meter = AverageMeter()
    model.eval()
    end = time.time()


    # First forward: count confidence threshold based on prior class ratio
    logger.info('<<<<<<<<<<<<<<<<< First forward Start <<<<<<<<<<<<<<<<<')

    global_cf_dict = [[] for _ in range(args.classes)]

    for i, (input, label) in enumerate(test_loader):

        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape

        if args.save_npy_or_png == 'npy':
            prediction = np.zeros((h, w, classes), dtype=float)
            for scale in scales:
                long_size = round(scale * base_size)
                new_h = long_size
                new_w = long_size
                if h > w:
                    new_w = round(long_size / float(h) * w)
                else:
                    new_h = round(long_size / float(w) * h)
                image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                prediction += scale_process(model, image_scale, classes, crop_h, crop_w, h, w, mean, std)
            prediction /= len(scales)  # (1024, 2048, 19) probas
            prediction_tensor = torch.from_numpy(prediction).cuda()
            prediction_tensor = prediction_tensor.type(torch.float16)
        else:
            prediction = np.squeeze(label.numpy())
            prediction_tensor = torch.squeeze(label.cuda())

        if args.temp_scaling:
            prediction_tensor = prediction_tensor / args.diagram_T

        prediction_tensor = torch.softmax(prediction_tensor, dim=2)

        pseudo_label = torch.argmax(prediction_tensor, dim=2)
        max_proba = torch.max(prediction_tensor, dim=2)[0]

        for j in range(args.classes):
            class_j_position = (pseudo_label == j)
            class_j_pseudo_probabilities = max_proba[class_j_position]  # len == class j number, already flattened

            toappend_numpy = class_j_pseudo_probabilities.cpu().numpy()

            toappend_list = toappend_numpy.tolist()
            global_cf_dict[j].extend(toappend_list)


        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)) and main_process(args):
            logger.info('First Forward: [{}/{}] '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    batch_time=batch_time))

        if args.save_npy_or_png == 'npy':
            check_makedirs(gray_folder)
            image_path, gt_label_path = data_list[i]
            image_name = image_path.split('/')[-1].split('.')[0]
            # import ipdb; ipdb.set_trace(context=20)
            # remove prefix in path
            if args.pseudo_data == 'cityscapes':
                prefix = "leftImg"
            elif args.pseudo_data == 'voc2012':
                prefix = "JPEGImages"
            elif args.pseudo_data == 'scannet' or 'gta5':
                prefix = "images"


            temp = image_path.split(prefix, 1)
            image_path = image_path.replace(temp[0], '')

            temp = gt_label_path.split("gtFine", 1)
            gt_label_path = gt_label_path.replace(temp[0], '')

            gray = np.float16(prediction)
            if args.pseudo_data == 'cityscapes':
                gray_path = os.path.join(gray_folder, image_name[:-11] + 'prediction.npy')
            elif args.pseudo_data == 'scannet' or 'gta5' or 'voc2012':
                gray_path = os.path.join(gray_folder, image_name + 'prediction.npy')
            temp2 = gray_path.split("pseudo_labels", 1)
            gray_path_in_list = gray_path.replace(temp2[0], '')
            np.save(gray_path, gray)
            if args.list_write == 'img_pse':
                list_writer.write(image_path + ' ' + gray_path_in_list + '\n')
            elif args.list_write == 'pse_gt':
                list_writer.write(gray_path_in_list + ' ' + gt_label_path + '\n')

    logger.info('Saving to confidence bank...')
    describ = args.npy_describ
    import subprocess
    cmd_line = "mkdir -p {0}".format('exp/npy_files/'+describ)
    subprocess.call(cmd_line.split())  # if this throws out errors, please mkdir before generate pseudo labels and comment this line (related to machine setup)
    for j in range(args.classes):
        global_cf_dict[j] = np.array(global_cf_dict[j])
        np.save('exp/npy_files/'+describ+'/class_'+str(j)+'.npy', global_cf_dict[j])
        logger.info("saving npy file of class_"+str(j))
    logger.info(describ)
    logger.info(args.prediction_list)
    logger.info('<<<<<<<<<<<<<<<<< First forward End <<<<<<<<<<<<<<<<<')


    if main_process(args):
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    list_writer.close()


# second forward
def second_forward(test_loader, data_list, model, classes, mean, std, base_size, crop_h, crop_w, scales,
                              gray_folder, list_path, logger, args):
    if main_process(args):
        logger.info('>>>>>>>>>>>>>>>> Generate pseudo labels >>>>>>>>>>>>>>>>')
    list_writer = open(list_path, 'a')
    batch_time = AverageMeter()
    model.eval()
    end = time.time()

    threshold_list, sample_ratio = get_threshold_and_sampling_ratio(args, logger)

    logger.info('<<<<<<<<<<<<<<<<< Second forward Start <<<<<<<<<<<<<<<<<')
    for i, (input, label) in enumerate(test_loader):

        input = np.squeeze(input.numpy(), axis=0)
        image = np.transpose(input, (1, 2, 0))
        h, w, _ = image.shape
        prediction_tensor = torch.squeeze(label.cuda())


        if args.temp_scaling:
            prediction_tensor = prediction_tensor / args.diagram_T

        prediction_tensor = torch.softmax(prediction_tensor, dim=2)

        # use obtained class-wise thresholds to generate initial pseudo labels by thresholding
        pseudo_label = torch.argmax(prediction_tensor, dim=2)
        max_proba = torch.max(prediction_tensor, dim=2)[0]

        for j in range(args.classes):
            threshold = threshold_list[j]
            class_j_pixel = (pseudo_label == j)
            class_j_confidence = max_proba.float() * class_j_pixel.float()
            class_j_confidence_invalid = (class_j_confidence < threshold) * class_j_pixel
            pseudo_label[class_j_confidence_invalid] = 255

        # use obtained sampling ratio to perform random sampling
        sampling_class = [x for x in range(args.classes)]

        sample_index = 0
        for j in sampling_class:

            class_j_over_threshold = (pseudo_label == j)
            class_j_over_threshold_num = class_j_over_threshold.sum()
            class_j_sample_num = class_j_over_threshold_num.float() * sample_ratio[sample_index]
            num_to_ignore = int(class_j_over_threshold_num - class_j_sample_num)
            sample_index += 1

            mask = torch.nonzero(class_j_over_threshold)    # class_j_over_threshold_num * 2, x y location
            random_indice = torch.randperm(mask.size(0))
            mask = mask[random_indice]                      # shuffle

            chosen_mask = mask[:num_to_ignore]      # where to set to ignore(255)
            pseudo_label[chosen_mask.t().chunk(chunks=2,dim=0)] = 255


        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)) and main_process(args):
            logger.info('Test: [{}/{}] '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(test_loader),
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)

        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]

        if args.pseudo_data == 'cityscapes':
            prefix = "leftImg"
        elif args.pseudo_data == 'scannet':
            prefix = "images"
        elif args.pseudo_data == 'voc2012':
            prefix = "JPEGImages"

        # remove prefix in path
        temp = image_path.split(prefix, 1)
        image_path = image_path.replace(temp[0], '')

        gray = np.uint8(pseudo_label.cpu().numpy())
        if args.pseudo_data == 'cityscapes':
            gray_path = os.path.join(gray_folder, image_name[:-11] + 'pseudo_label.png')
        elif args.pseudo_data == 'scannet':
            gray_path = os.path.join(gray_folder, image_name + 'pseudo_label.png')
        elif args.pseudo_data == 'voc2012':
            gray_path = os.path.join(gray_folder, image_name + 'pseudo_label.png')
        temp2 = gray_path.split("pseudo_labels", 1)
        gray_path_in_list = gray_path.replace(temp2[0], '')
        cv2.imwrite(gray_path, gray)
        list_writer.write(image_path + ' ' + gray_path_in_list + '\n')
    logger.info('<<<<<<<<<<<<<<<<< Second forward End <<<<<<<<<<<<<<<<<')
    logger.info('Saving to'+ args.save_pseudo_label_path)
    if main_process(args):
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    list_writer.close()

