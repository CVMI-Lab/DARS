import os
import os.path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import copy

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            # if len(line_split) != 2:
            #     raise (RuntimeError("Image list file read line error : " + line + "\n"))
            if len(line_split) == 2:
                # raise (RuntimeError("Image list file read line error : " + line + "\n")) 
                image_name = os.path.join(data_root, line_split[0])
                label_name = os.path.join(data_root, line_split[1])
            if len(line_split) == 1:
                image_name = os.path.join(data_root, line_split[0])
                label_name = image_name
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list

def make_dataset3(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            if len(line_split) != 3:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            pse_label_name = os.path.join(data_root, line_split[1])
            gt_label_name = os.path.join(data_root, line_split[2])
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, pse_label_name, gt_label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        return image, label

class SemData_soft(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        if image is None:
            raise (RuntimeError("Image does not exist" + image_path + "\n"))
        if self.transform is not None:

            label_list = None
            # import ipdb
            # ipdb.set_trace(context=20)
            for i in range(19):
                label = cv2.imread(label_path+str(i) +'.png', cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
                label = np.expand_dims(label,axis=0)    # 1HW
                if label_list is None:
                    label_list = label
                else:
                    label_list = np.concatenate((label_list, label), axis=0)    # finally, label_list 19HW

            # print('in dataset.py {}'.format(label_list.shape))
            image, soft_label = self.transform(image, label_list)

            # print(soft_label.max())
        return image, soft_label


class SemData_img_pse_gt(Dataset):
    # previously wrong(transform twice, lead to mismatch) now used for inpainting
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset3(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path, gt_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            label = np.expand_dims(label, axis=0)
            gt = np.expand_dims(gt, axis=0)
            label_list = np.concatenate((label, gt), axis=0)
            # print("label_list.shape ={}".format(label_list.shape))
            image, labels = self.transform(image, label_list)
            # import ipdb
            # ipdb.set_trace(context=20)
            # print("labels.shape ={}".format(labels.shape))
            label, gt = labels

            # 255 -> 19
            label_255 = (label == 255)
            label[label_255] = 19
            # label = label / 18.0

            gt_255 = (label == 255)
            gt[gt_255] = 19

            # print("img.max ={}".format(image.max()))
            # print("img.min ={}".format(image.min()))
            # print("gt.shape ={}".format(gt.shape))
        return image, label, gt


class SemData_npy(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = np.load(label_path)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, _ = self.transform(image)
            label = torch.from_numpy(label)
        return image, label


class SemData_logits(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        image = np.load(image_path)  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            label, _ = self.transform(label)
            image = torch.from_numpy(image)
        return image, label


class SemData2(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, label=True, base_transform=None, require_label=False):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.base_transform = base_transform
        self.label = label
        self.require_label = require_label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        # import copy
        # image_original = copy.deepcopy(image)
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.label:
            if self.transform is not None:
                image, label = self.transform(image, label)
            return image, label
        else:
            if self.base_transform is not None and self.require_label is False:
                image, _ = self.base_transform(image, None)
                return image
            if self.base_transform is not None and self.require_label is True:
                image, label = self.base_transform(image, label)
                return image, label


class SemData3(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, label=True, base_transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.base_transform = base_transform
        self.label = label

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W
        import copy
        image_original = copy.deepcopy(image)
        # if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
        #     raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.label:
            if self.transform is not None:
                image, label = self.transform(image, label)
            mydict = {}
            mydict["image"] = image
            mydict["label"] = label
            mydict["image_path"] = image_path
            mydict["label_path"] = label_path
            mydict["index"] = index
            return mydict
        else:
            if self.base_transform is not None:
                image_original, _ = self.base_transform(image_original, None)
                # print("index = {}".format(index))   # todo
            return image_original
