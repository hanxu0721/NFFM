#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:09:29 2019

@author: hanxu8
"""
import numpy as np
import tensorflow as tf

#获取上述index的所有数据              
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
def get_input2(data_batch, field):
    idx = 0
    sparse_dict = {} 
    field_cnt = len(field)
    #indices,values,shapes = [[]] * field_cnt, [[]] * field_cnt, [[0,0]] * field_cnt
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
