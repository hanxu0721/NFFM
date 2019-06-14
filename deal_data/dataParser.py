#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:09:29 2019

@author: hanxu8
"""
import numpy as np
import pandas as pd
import tensorflow as tf


## 获取feature字典,对每一个field设置一个None值的index
def get_dict(path_list):
    tc_dic = {}
    userFeature_dict = {}
    for path in path_list:
        with open(path,'r') as f:
            for line in f:
                line = line.strip('\n').split('\t')
                features = line[1]
                each_list = features.split(',')
                for i in field:
                    for fea in each_list:
                        if fea.find(i)>=0 and fea not in userFeature_dict[i]:
                            userFeature_dict[i][fea]=tc_dic[i]
                            tc_dic[i] += 1
    for i in field:
        userFeature_dict[i][i+'_None']=tc_dic[i]
    return userFeature_dict



##将feature处理成index的格式，共有15个field
# 0	86	25	80	21	2,22,28,30,37,44	70	234	31	
# 0	194	25	201	188	37,97,115,138	70	151,162,163	31
# 0	226	25	223	218	206,207,208,37,44,227	70	234	31

def fea2index(path_list, data_dir, field, userFeature_dict):
    for i ,path in enumerate(path_list):
        output = open(data_dir+'/'+'fea2index'+'/'+str(i),'w')
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

##获取上述index的所有数据              
def decode(path):
    if len(path) == "":
        raise ValueError("data is empty")
    x = []
    y = []
    with open(path,'r') as f:
        for line in f:
            lis = line.strip('\n').split('\t')
            label = lis[0]
            fea = lis[1:len(lis)]
            x.append(fea)
            y.append(label)
    return x,y

def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

##获取batch数据
def get_batch(x, y, batch_size, index):
    start = index * batch_size
    end = (index+1) * batch_size
    end = end if end < len(y) else len(y)
    data_batch = x[start:end]
    y = [[y_] for y_ in y[start:end]]
    return data_batch,y

def get_input(data_batch, field):
    idx = 0
    sparse_dict = {} 
    field_cnt = len(field)
    indices,values,shapes = [], [], []
    for tag in field:
        indices.append([])
        values.append([])
        shapes.append([0,0])
    
    for fea in data_batch:
        for i in range(field_cnt):
            x = fea[i].split(',')
            x = [int(item) for item in x]
            if shapes[i][1] < len(x):
                shapes[i][1] = len(x)
            values[i] += x
            for j in xrange(len(x)):
                indices[i].append([idx,j])
            shapes[i][0] += 1
        idx += 1
    for i, j in enumerate(field):
        sparse_dict[j] = (indices[i], values[i], shapes[i])
        
    return sparse_dict
