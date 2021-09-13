#!/bin/sh
# usage:
# sh tool/train.sh cityscapes config-1.yaml ex200
PYTHON=python3

dataset=$1
cfg=$2
exp_name=$3

exp_dir=exp/${exp_name}
result_dir=${exp_dir}/result
model_dir=${exp_dir}/model
config=config/${dataset}/${cfg}
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
mkdir -p ${exp_dir}
mkdir -p ${model_dir}
cp tool/train.sh tool/train.py tool/train_func.py tool/test.py model/pspnet.py model/deeplabv2.py ${config} ${exp_dir}


export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export PYTHONPATH=./
export KMP_INIT_AT_FORK=FALSE
$PYTHON -u ${exp_dir}/train.py --config=${exp_dir}/${cfg} exp_name ${exp_name} 2>&1 | tee ${exp_dir}/train-$now.log



