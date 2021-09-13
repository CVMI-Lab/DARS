#!/bin/sh
# sh tool/scripts/train_psp50_voc_split1464_crop321.sh

# --- Round 0: train with only labeled data
sh tool/train.sh voc2012 train_psp50_voc_split1464_crop321_round0.yaml train_psp50_voc_split1464_crop321_round0

# --- Pseudo label Generation for Round 1
cp exp/train_psp50_voc_split1464_crop321_round0/model/train_epoch_50.pth initmodel/train_psp50_voc_split1464_crop321_round0.pth
sh tool/gen_pseudo_label.sh voc2012 gen_pseudo_label_psp50_voc_split1464_crop321_for_round1.yaml ex_gen
rm dataset/voc2012/pseudo_labels/psp50_voc_split1464_crop321_for_round1/1/*.npy  # rm logit npy files to save disk space
rm -r exp/npy_files/npy_psp50_voc_split1464_crop321_for_round1 # rm confidence npy files to save disk space

# --- Round 1: train with labeled data and unlabeled data (equipped with generated pseudo labels)
sh tool/train.sh voc2012 train_psp50_voc_split1464_crop321_round1.yaml train_psp50_voc_split1464_crop321_round1

# --- Pseudo label Generation for Round 2
cp exp/train_psp50_voc_split1464_crop321_round1/model/train_epoch_100.pth initmodel/train_psp50_voc_split1464_crop321_round1.pth
sh tool/gen_pseudo_label.sh voc2012 gen_pseudo_label_psp50_voc_split1464_crop321_for_round2.yaml ex_gen
rm dataset/voc2012/pseudo_labels/psp50_voc_split1464_crop321_for_round2/1/*.npy  # rm logit npy files to save disk space
rm -r exp/npy_files/npy_psp50_voc_split1464_crop321_for_round2 # rm confidence npy files to save disk space

# --- Round 2: train with labeled data and unlabeled data (equipped with generated pseudo labels)
sh tool/train.sh voc2012 train_psp50_voc_split1464_crop321_round2.yaml train_psp50_voc_split1464_crop321_round2