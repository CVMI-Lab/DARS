#!/bin/sh
# usage:
# sh tool/gen_pseudo_label.sh cityscapes config_gen.yaml ex_test

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
cp tool/gen_pseudo_label.sh tool/gen_pseudo_label.py tool/test.py  model/pspnet.py ${config} ${exp_dir}

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export PYTHONPATH=./
export KMP_INIT_AT_FORK=FALSE
$PYTHON -u ${exp_dir}/gen_pseudo_label.py --config=${exp_dir}/${cfg}  2>&1 | tee ${exp_dir}/train-$now.log
