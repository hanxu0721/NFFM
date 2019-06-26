#!/bin/bash
cur_dir=`pwd`
source ${cur_dir}/model.conf
source ${cur_dir}/run.conf

num_ins=1000
num_runs=1
# 超参
info="max_epoch=${max_epochs} batch_size=${batch_size} factor_size=${factor_size} drop_out=${drop_out} l2_reg=${l2_reg}"

# Deep隐藏层
deep="deep_layers=${deep_layers}"

# learning
learning="optimizer=${optimizer} learning_rate=${learning_rate}"

echo -e "${info} ${deep} ${learning} "

train(){
    hdfs dfs -mkdir ${model};  hdfs dfs -chmod 777 ${model};  hdfs dfs -rmr ${model}/*
    hdfs dfs -mkdir ${log_dir};hdfs dfs -chmod 777 ${log_dir};hdfs dfs -rmr ${log_dir}/*
    TensorFlow_Submit  \
    --appName="${person}_${app_name}"  \
    --archives=hdfs://ns3-backup/dw_ext/ad/person/wenwei6/nn/Python_2.7.13_TF_1.2.0.zip#Python \
    --files=./nffm_data_parser.py,./train_hadoop.py,./nffm.py,./config.py \
    --worker_memory=${worker_memory} \
    --worker_cores=2  \
    --num_worker=${num_worker}  \
    --num_ps=${num_ps}   \
    --ps_memory=${ps_memory} \
    --appType=TensorFlow \
    --mode_local=true \
    --tensorboard=true \
    --data_dir=${instance} \
    --log_dir=${log_dir}/  \
    --train_dir=${model}/ \
    --command=Python/bin/python train_hadoop.py num_runs=${num_runs} num_ins=${num_ins} field_size=${field_size} ${info} ${deep} ${learning}
}
train

