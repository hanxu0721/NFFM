#!/bin/bash

HDFS_DIR=/dw_ext/ad/ads_dm/hanxu8/

table_name="user_feature_label_data"
sql_feature="
drop table if exists ${table_name};
CREATE TABLE IF NOT EXISTS ${table_name}
(
    uid        string    comment 'uid',
    label      string    comment '标签',
    feature    string    comment  '特征'
)
PARTITIONED BY(type String,pkg_id STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '${HDFS_DIR}/${table_name}';
ALTER TABLE ${table_name} SET SERDEPROPERTIES('serialization.null.format'=''); 
"
#hive -e"$sql_feature"

table_name="user_feature_data"
sql_feature="
drop table if exists ${table_name};
CREATE TABLE IF NOT EXISTS ${table_name}
(
    uid        string    comment 'uid',
    feature    string    comment  '特征'
)
PARTITIONED BY(type STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '${HDFS_DIR}/${table_name}';
ALTER TABLE ${table_name} SET SERDEPROPERTIES('serialization.null.format'=''); 
"
#hive -e"$sql_feature"

table_name="pkg_id_user_label_data"
sql_feature="
drop table if exists ${table_name};
CREATE TABLE IF NOT EXISTS ${table_name}
(
    pkg_id     string    comment '包id',
    uid        string    comment 'uid',
    label      string    comment '标签'
)
PARTITIONED BY(type STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '${HDFS_DIR}/${table_name}';
ALTER TABLE ${table_name} SET SERDEPROPERTIES('serialization.null.format'=''); 
"
#hive -e"$sql_feature"

#hive -e"load data inpath '/dw_ext/ad/person/hanxu8/DL/nffm/train_user_feature' overwrite into table user_feature_data partition (type='train')"
#hive -e"load data inpath '/dw_ext/ad/person/hanxu8/DL/nffm/test_user_feature' overwrite into table user_feature_data partition (type='test')"
#hive -e"load data inpath '/dw_ext/ad/person/hanxu8/DL/nffm/train_data1' overwrite into table pkg_id_user_label_data partition (type='train')"
#hive -e"load data inpath '/dw_ext/ad/person/hanxu8/DL/nffm/test_data1' overwrite into table pkg_id_user_label_data partition (type='test')"
hql="
set mapreduce.job.reduces=1000;
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nostrick;
insert overwrite table user_feature_label_data
partition(type='test',pkg_id)
select a.uid , a.label, b.feature, a.pkg_id as pkg_id
from
   (select pkg_id, uid, 'none' as label from pkg_id_user_label_data where type='test')a
join
   (select uid, feature from user_feature_data where type='test')b
on a.uid = b.uid
group by a.uid , a.label, b.feature, a.pkg_id
"
hive -e"$hql"


table_name="feature_label_data_dealed"
sql_feature="
drop table if exists ${table_name};
CREATE TABLE IF NOT EXISTS ${table_name}
(
    label      string    comment '标签',
    feature    string    comment '特征'
)
PARTITIONED BY(pkg_id STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
STORED AS TEXTFILE
LOCATION '${HDFS_DIR}/${table_name}';
ALTER TABLE ${table_name} SET SERDEPROPERTIES('serialization.null.format'=''); 
"
#hive -e"$sql_feature"
hql="
add file deal.py;
insert overwrite table feature_label_data_dealed
partition (pkg_id='218837855')
select transform (a.label,a.feature)
using 'python deal.py'
as label, feature
from
   (select label, feature from user_feature_label_data where pkg_id='218837855')a
"
#hive -e"$hql"
