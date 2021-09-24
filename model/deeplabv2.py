"""
Taken and adapted from:
https://github.com/hfslyc/AdvSemiSeg/blob/master/model/deeplab.py

Adversarial Learning for Semi-supervised Semantic Segmentation
Wei-Chih Hung, Yi-Hsuan Tsai, Yan-Ting Liou, Yen-Yu Lin, and Ming-Hsuan Yang
Proceedings of the British Machine Vision Conference (BMVC), 2018.

"""

import os
import torch.nn as nn, torch.nn.functional as F
import math
import torch
import numpy as np
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.models import resnet

affine_par = True


_RESNET_101_DEEPLAB_COCO_URL = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
_RESNET_101_IMAGENET_URL = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

def freeze_bn_module(m):
    """ Freeze the module `m` if it is a batch norm layer.
    :param m: a torch module
    :param mode: 'eval' or 'no_grad'
    """
    classname = type(m).__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out


class ResNetDeepLab(nn.Module):
    # BLOCK_SIZE = (16, 16)
    BLOCK_SIZE = (1, 1)

    def __init__(self, block, layers, num_classes, criterion=nn.CrossEntropyLoss(ignore_index=255)):
        self.inplanes = 64
        super(ResNetDeepLab, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.criterion = criterion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x, y=None, sup_loss_method='CE'):
        in_shape = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        # The implementation of this network came from the repo of
        # Hung et al. at https://github.com/hfslyc/AdvSemiSeg, one of the works we compare against.
        # They did their upsampling in their training script, where we do it in our network.
        # They use `align_corners=True` when the PyTorch version is >= 0.4:
        # https://github.com/hfslyc/AdvSemiSeg/blob/841d546c927605747594be726622363bf781cefb/train.py#L298
        # We remain consistent with this.
        # Torchvision has however decided to go with `align_corners=False`. Its probably more correct.
        # Furthermore, their experiments showed that it makes little difference in terms of performance:
        # https://github.com/pytorch/vision/issues/1708
        x = F.interpolate(x, size=in_shape[2:4], mode='bilinear', align_corners=True)

        if y is not None:
            main_loss = self.criterion(x, y)
            aux_loss = main_loss
        else:
            main_loss = None
            aux_loss = None

        if main_loss is not None:
            # return x.max(1)[1], main_loss, aux_loss
            return (x, x), main_loss, aux_loss
        else:
            return x

    def pretrained_parameters(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def pretrained_layers(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        return b


    def new_parameters(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def new_layers(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5)

        return b

    def freeze_batchnorm(self):
        self.apply(freeze_bn_module)


def Resnet101_deeplab(num_classes=21, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, pretrain_data='?'):
    model = ResNetDeepLab(Bottleneck, [3, 4, 23, 3], num_classes, criterion=criterion)
    if pretrained:
        # saved_state_dict = load_state_dict_from_url(_RESNET_101_DEEPLAB_COCO_URL)
        if pretrain_data == 'coco':
            model_path = './initmodel/resnet101COCO-41f33a49.pth'
            print('loaded deeplabv2 resnet101 coco weights')
        elif pretrain_data == 'imagenet':
            model_path = './initmodel/resnet101-5d3b4d8f.pth'
            print('loaded deeplabv2 resnet101 imagenet weights')
        saved_state_dict = torch.load(model_path)
        new_params = model.state_dict().copy()
        for name, param in new_params.items():
            if name in saved_state_dict and param.size() == saved_state_dict[name].size():
                new_params[name].copy_(saved_state_dict[name])
        model.load_state_dict(new_params)
        # import ipdb;ipdb.set_trace(context=20)
    return model


def _load_state_into_model(model, state_dict, verbose=False):
    new_params = model.state_dict().copy()
    for name, param in new_params.items():
        if name in state_dict:
            if param.size() == state_dict[name].size():
                new_params[name].copy_(state_dict[name])
            else:
                if verbose:
                    print('{} -> {}'.format(state_dict[name].shape, new_params[name].shape))
        else:
            if verbose:
                print('Could not find {}'.format(name))
    model.load_state_dict(new_params)