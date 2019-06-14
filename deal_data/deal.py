#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 14:26:02 2019

@author: hanxu8
"""
import tensorflow as tf
import os

def get_dict(path_list):
    tc_dic = {}
    userFeature_dict = {}
    for i in field:
        tc_dic[i]=0
        userFeature_dict[i]={}
        userFeature_dict[i][i+'_None']=tc_dic[i]
    for path in path_list:
        with open(path,'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                features = line[2]
                each_list = features.split(',')
                for i in field:
                    for fea in each_list:
                        if fea.find(i)>=0 and fea not in userFeature_dict[i]:
                            tc_dic[i] += 1
                            userFeature_dict[i][fea]=tc_dic[i]
    print ('userFeature_dict={}'.format(userFeature_dict))
    num = {}
    for i in field:
        num[i] = len(list(userFeature_dict[i].keys()))
    print ('\n')
    print ('total_fea_dict={}'.format(num))
    return userFeature_dict


def fea2index(path_list, data_dir, field, userFeature_dict):
    for i ,path in enumerate(path_list):
        output = open(data_dir+'/'+str(i),'w')
        with open(path,'r') as f:
            for line in f:
                total_fea = []
                line = line.strip('\n').split('\t')
                label = line[1]
                fea = line[2].split(',')
                for i in field:
                    index = []
                    for j in fea:
                        if j.find(i)>=0:
                            index.append(str(userFeature_dict[i][j]))
                    if len(index) == 0:
                        index.append(str(userFeature_dict[i][i+'_None']))
                    total_fea.append(index)
                output.write(label)
                for fea in total_fea:
                    output.write('\t{}'.format(','.join(fea)))
                output.write('\n')
if __name__ == '__main__':
    path='/data3/ads_dm/hanxu8/nFFM/deal_data/pkg_id=218837855'
    parent_path = '/data3/ads_dm/hanxu8/nFFM/deal_data/fea2index'
    filenames = tf.gfile.ListDirectory(path)
    filenames = [x for x in filenames]
    filenames = [os.path.join(path, x) for x in filenames]
    filenames = [x for x in filenames if tf.gfile.Exists(x)]
    ins_list = []
    field = ['AGE','GEND','PROV','CITY','BHV','PLAT','LIFE','LOGIN','CUST','ACTDAY','ACTMID'
         ,'ACTCNT','IC','EXPOCNT','MOB']
    for item in filenames:
        ins_list.append(item)
    
    userFeature_dict = get_dict(ins_list)
    #fea2index(ins_list, parent_path, field, userFeature_dict)
