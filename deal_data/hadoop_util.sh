#!/bin/bash
HADOOP_BIN=`which hadoop`
echo util.sh $HADOOP_BIN
HADOOP=$HADOOP_BIN
mydir=`pwd`

function myexit() {
    if [ $? != 0 ]
    then
        echo "exit, fatal"
        exit 1
    fi
}
test_file_rmr() {
    ${HADOOP_BIN} fs -test -e $1
    if [ $? -ne "0" ];then
        echo "[i see, hadoop output $1 do not exist]"
    else
        ${HADOOP_BIN} fs -rmr $1
    fi
    myexit
}
base_call_fix() {
    local out=$2
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -D stream.map.output.field.separator="	" \
        -D stream.num.map.output.key.fields=1 \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
base_call_sep() {
    local out=$2
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
base_call_cachefile() {
    local out=$2
    local cachefile=$9
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -cacheFile $cachefile#appdict \
        -jobconf mapred.output.compress=true \
        -jobconf mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
base_call_multi() {
    local out=$2
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -libjars ${mydir}/multiout.jar \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -outputformat adsfanstop.multiout \
        -jobconf mapred.output.compress=true \
        -jobconf mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
base_call_nozip() {
    local out=$2
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
base_call() {
    local out=$2
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -jobconf mapred.output.compress=true \
        -jobconf mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
base_call_sort2_weibo() {
    local out=$2
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -D stream.map.output.field.separator="	" \
        -D stream.num.map.output.key.fields=2 \
        -D map.output.key.field.separator="	" \
        -D num.key.fields.for.partition=1 \
        -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
base_call_sort2() {
    local out=$2
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -D stream.num.map.output.key.fields=2 \
        -D num.key.fields.for.partition=1 \
        -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
base_call_sort2_fix() {
    local out=$2
    test_file_rmr $out
    ${HADOOP_BIN} jar /usr/local/hadoop-2.7.3/share/hadoop/tools/lib/hadoop-streaming-2.7.3.jar  \
        -D stream.map.output.field.separator="	" \
        -D stream.num.map.output.key.fields=2 \
        -D map.output.key.field.separator="	" \
        -D mapred.text.key.partitioner.options=-k2,2 \
        -D mapred.text.key.comparator.options="-k2,2n" \
        -D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
        -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
        -input $1 \
        -output $out \
        -mapper "$3" \
        -reducer "$4" \
        -file $5 \
        -jobconf mapred.job.name=$6 \
        -jobconf mapred.map.tasks=$7 \
        -jobconf mapred.job.map.capacity=$7 \
        -jobconf mapred.reduce.tasks=$8 \
        -jobconf mapred.job.reduce.capacity=$8 
    myexit
    ${HADOOP_BIN} fs -dus $out
}
same_haoop_cp() {
    root_src=$1
    root_dest=$2
    HADOOP_BIN=$3
    ${HADOOP_BIN} fs -rmr ${root_dest}
    ${HADOOP_BIN} distcp  -su fcr,SaK2VqfEDeXzKPor hdfs://nmg01-khan-hdfs.dmop.baidu.com:54310/${root_src} hdfs://nmg01-khan-hdfs.dmop.baidu.com:54310/${root_dest}
}
khan2mulan() {
    root_mulan=$1
    root_khan=$2
    ${HADOOP_BIN} fs -rmr ${root_mulan}
    ${HADOOP_BIN} distcp  -su fcr,SaK2VqfEDeXzKPor hdfs://nmg01-khan-hdfs.dmop.baidu.com:54310/${root_khan} hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310/${root_mulan}
}

online2online() {
    hadoop_src=$1
    hadoop_dst=$2
    ${HADOOP_BIN} fs -rmr ${hadoop_dst}
    ${HADOOP_BIN} distcp -su app,app -du fcr,SaK2VqfEDeXzKPor ${hadoop_src} ${hadoop_dst}
}

function testexit() {
    ls
    myexit
    mkdir kk/gg
    myexit
    echo 'myexit after'
}
split_str() {
    line=$1
    delim=$2
    index=$3
    arr=${line//${delim}/ }
    echo ${arr[1]}
    #echo ${ line[${index}] }
    myexit
}
mycut() {
    line=$1
    delim=$2
    index=$3
    echo $line | cut -d ${delim} -f${index}
    myexit
}
wait_command_over() {
    while true
    do
        if [ -f $1 ];then
            return
        fi
        sleep 30
    done
    myexit
}
get_hadoop() {
    if [ -f "$1" ];then
        echo "mv" $1
        echo "and get file from hadoop"
        mv $1 ${1}.bk
    fi
    hadoop fs -get $2 $1
    myexit
}
getmerge_hadoop() {
    echo $1 $2
    if [ -f "$1" ];then
        echo "mv" $1
        mv $1 ${1}.bk
    fi
    hadoop fs -getmerge $2 $1
    myexit
}
test_local_file_copy() {
    if [ -f "$1" ];then
        echo "delete" $1
        echo "and get file from hadoop"
        mv $1 ${1}.bk
    fi
    hadoop fs -get $2 $1
    myexit
}
mv_file_hadoop() {
    hadoop fs -test -e $1
    if [ $? -ne "0" ];then
        echo "[i see, hadoop output $1 do not exist]"
    else
        hadoop fs -rmr ${1}.bk
        hadoop fs -mv $1 ${1}.bk
    fi
    myexit
}

up_file2hadoop() {
    mv_file_hadoop $2
    hadoop fs -put $1 $2
    myexit
}
mvfile() {
    echo $1 $2
    if [ -f "$1" ];then
        echo "mv" $1
        mv $1 $2
    fi
    myexit
}

###same_haoop_cp "/app/ecom/fcr/adapp/applist/20150322" "/app/ecom/fcr/adapp/applist/20150323" "/home/qiaofangjie/hadoop/hadoop-client-nmg/hadoop/bin/hadoop"
