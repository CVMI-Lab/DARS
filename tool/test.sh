#!/bin/sh

#PYTHON=/mnt/backup2/home/yczhang7/segahsj/anaconda3/envs/cuda10.2/bin/python3.7
PYTHON=python3
#dataset=cityscapes
dataset=voc2012
exp_name=$1
exp_dir=exp/${exp_name}
result_dir=${exp_dir}/result
config=config/${dataset}/config_hu_0.60_gen.yaml
now=$(date +"%Y%m%d_%H%M%S")

mkdir -p ${result_dir}
mkdir -p ${exp_dir}
cp tool/test.sh tool/test.py ${config} ${exp_dir}
#cp tool/test.sh tool/my_train2_task2.py ${config} ${exp_dir}
#cp tool/test.sh tool/only_label.py ${config} ${exp_dir}

export PYTHONPATH=./
export KMP_INIT_AT_FORK=FALSE
#$PYTHON -u ${exp_dir}/my_train2.py --config=${config}  2>&1 | tee ${model_dir}/train-$now.log
#$PYTHON -u ${exp_dir}/my_train2_task2.py --config=${config}  2>&1 | tee ${result_dir}/test-$now.log
$PYTHON -u ${exp_dir}/test.py --config=${config}  2>&1 | tee ${result_dir}/test-$now.log
#$PYTHON -u ${exp_dir}/only_label.py --config=${config}  2>&1 | tee ${result_dir}/test-$now.log
