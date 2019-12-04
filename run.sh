#!/bin/bash

cur_dir=`pwd`
#source "${cur_dir}/src/data.conf"

function training() {
    source ${cur_dir}/src/run.conf
    cd ${cur_dir}/src
    nohup sh -x run.sh ${cur_time} ${person} > ${cur_dir}/log/${today}.${cur_time}.log.training 2>&1
}

training
