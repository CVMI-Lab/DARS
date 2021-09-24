#!/bin/sh
# sh tool/scripts/train_deeplabv2_cityscapes_split8_crop361.sh

# --- Round 0: train with only labeled data
sh tool/train.sh cityscapes train_deeplabv2_cityscapes_split8_crop361_round0.yaml train_deeplabv2_cityscapes_split8_crop361_round0

# --- Pseudo label Generation for Round 1
cp exp/train_deeplabv2_cityscapes_split8_crop361_round0/model/train_epoch_200.pth initmodel/train_deeplabv2_cityscapes_split8_crop361_round0.pth
sh tool/gen_pseudo_label.sh cityscapes gen_pseudo_label_deeplabv2_cityscapes_split8_crop361_for_round1.yaml ex_gen
rm dataset/cityscapes/pseudo_labels/deeplabv2_cityscapes_split8_crop361_for_round1/1/*.npy  # rm logit npy files to save disk space
rm -r exp/npy_files/npy_deeplabv2_cityscapes_split8_crop361_for_round1 # rm confidence npy files to save disk space

# --- Round 1: train with labeled data and unlabeled data (equipped with generated pseudo labels)
sh tool/train.sh cityscapes train_deeplabv2_cityscapes_split8_crop361_round1.yaml train_deeplabv2_cityscapes_split8_crop361_round1

# --- Pseudo label Generation for Round 2
cp exp/train_deeplabv2_cityscapes_split8_crop361_round1/model/train_epoch_400.pth initmodel/train_deeplabv2_cityscapes_split8_crop361_round1.pth
sh tool/gen_pseudo_label.sh cityscapes gen_pseudo_label_deeplabv2_cityscapes_split8_crop361_for_round2.yaml ex_gen
rm dataset/cityscapes/pseudo_labels/deeplabv2_cityscapes_split8_crop361_for_round2/1/*.npy  # rm logit npy files to save disk space
rm -r exp/npy_files/npy_deeplabv2_cityscapes_split8_crop361_for_round2 # rm confidence npy files to save disk space

# --- Round 2: train with labeled data and unlabeled data (equipped with generated pseudo labels)
sh tool/train.sh cityscapes train_deeplabv2_cityscapes_split8_crop361_round2.yaml train_deeplabv2_cityscapes_split8_crop361_round2