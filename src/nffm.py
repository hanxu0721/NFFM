#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:15:07 2019

@author: hanxu8
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import itertools
from sklearn.metrics import roc_auc_score


class NFM:
    def __init__(self, 
                 field_size,
                 embedding_size,
                 total_fea_dict,
                 loss_type,
                 deep_layers=[20, 20],
                 batch_norm=0,
                 batch_norm_decay=0.995,
                 l2_reg=0.0,
                 eval_metric=roc_auc_score,
                 deep_layers_activation=tf.nn.relu):
        
        
        self.field_size = field_size
        self.total_fea_dict = total_fea_dict  ##每个field里的特征数
        #self.total_fea_sizes = sum(self.total_fea_dict.values()) ##特征总数，不分field
        self.total_features = list(self.total_fea_dict.keys())
        self.embedding_size = embedding_size
        self.loss_type = loss_type
        self.deep_layers  = deep_layers
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.l2_reg = l2_reg
        self.eval_metric = eval_metric
        self.deep_layers_activation = deep_layers_activation
        
        ## define params
        self.features = {}
        for key in self.total_fea_dict:
            self.features[key] = tf.sparse_placeholder(tf.int64, name = 'features')
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")  # None * 1
        self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
        self.train_phase = tf.placeholder(tf.bool, name="train_phase")
        self.weights = self._initialize_weights()
    
    def _initialize_weights(self):
        weights = dict()
        # lr layer
        weights["lr_embeddings"] = {}
        for key in self.total_fea_dict:
            weights["lr_embeddings"][key] = tf.Variable(
                tf.truncated_normal([self.total_fea_dict[key], 1], 0.0, 0.0001),
                name='lr_embeddings')
            
        weights["lr_bias"] = {}
        for key in self.total_fea_dict:
            weights["lr_bias"][key] = tf.Variable(
                tf.random_uniform([1], 0.0, 1.0), name="lr_bias")
        #ffm layer    
        weights["ffm_embedding"]={}
        for key in self.total_fea_dict:
            weights["ffm_embedding"][key] = tf.Variable(
                tf.truncated_normal([self.total_fea_dict[key], self.embedding_size * self.field_size], 0.0, 0.0001),
                name=key + 'embeddings')

        # deep layer
        num_layer = len(self.deep_layers)
        input_size = 0
        #features = self.total_features
        for (i1, i2) in itertools.combinations(list(range(0, len(self.total_features))), 2):
            #c1, c2 = features[i1], features[i2]
            input_size += 1
        input_size *= self.embedding_size
        #input_size = self.total_field_size * (self.total_field_size - 1) / 2 * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        #print input_size,glorot
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32) 
            
        input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
    

 
    def linear(self):
        with tf.name_scope("lr_layer"):
            y_linear_order = {}
            y_linear_sum = []
            for key in self.total_fea_dict:
                y_linear_order[key] = tf.nn.embedding_lookup_sparse(self.weights["lr_embeddings"][key], 
                                                               self.features[key], None, combiner = 'sum')+ self.weights["lr_bias"][key] #None*1
                y_linear_sum.append(y_linear_order[key])
            y_linear_sum = tf.reduce_sum(y_linear_sum, axis = 0)
            return y_linear_sum
        
    def FFM(self):
        with tf.name_scope("ffm_layer"):
            embed_var_raw_dict = {}
            embed_var_dict = {}
            for key in self.total_fea_dict:
                embed_var_raw = tf.nn.embedding_lookup_sparse(self.weights['ffm_embedding'][key], self.features[key], None, combiner = 'mean')
                embed_var_raw_dict[key] = tf.reshape(embed_var_raw, [-1, self.field_size, self.embedding_size]) #None*f*k
            for (i1, i2) in itertools.combinations(list(range(0, len(self.total_features))), 2):
                c1, c2 = self.total_features[i1], self.total_features[i2]
                embed_var_dict.setdefault(c1, {})[c2] = embed_var_raw_dict[c1][:, i2, :] # None * k
                embed_var_dict.setdefault(c2, {})[c1] = embed_var_raw_dict[c2][:, i1, :] # None * k
            x_mat = []
            y_mat = []
            input_size = 0
            for (c1, c2) in itertools.combinations(embed_var_dict.keys(), 2):
                input_size += 1
                x_mat.append(embed_var_dict[c1][c2]) #input_size * None * k
                y_mat.append(embed_var_dict[c2][c1]) #input_size * None * k
            x_mat = tf.transpose(x_mat, perm=[1, 0, 2]) # None * input_size * k
            y_mat = tf.transpose(y_mat, perm=[1, 0, 2]) # None * input_size * k
            x = tf.multiply(x_mat, y_mat)
            deep_input = tf.reshape(x, [-1, input_size * self.embedding_size])
            y_ffm_order = tf.reduce_sum(deep_input, axis = 1) # None * 1
            y_ffm_order = tf.reshape(y_ffm_order,[-1,1])
            return deep_input, y_ffm_order,self.weights['ffm_embedding']
    
    def deep(self,deep_input):
        with tf.name_scope("deep_layer"):
            y_deep = deep_input
            y_deep = tf.nn.dropout(y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                y_deep = tf.matmul(y_deep, self.weights["layer_%d" % i])
                #self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    y_deep = self.batch_norm_layer(y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                y_deep = self.deep_layers_activation(y_deep)
                y_deep = tf.nn.dropout(y_deep, self.dropout_keep_deep[1+i])
        
            deep_out = tf.add(tf.matmul(y_deep, self.weights["concat_projection"]), self.weights["concat_bias"])
            return deep_out
    
    def forward(self):
        with tf.name_scope("forward"):
            lr_out = self.linear()
            deep_input,ffm_out,weights = self.FFM()
            deep_out = self.deep(deep_input)
            out = tf.add(lr_out,ffm_out)
            out = tf.add(out,deep_out)
            y_predict = tf.nn.sigmoid(out)
            return y_predict
    
    def loss(self):
        with tf.name_scope("loss"):
            out = self.forward()
        #self.loss = tf.losses.log_loss(self.label, out)
            if self.loss_type == "log_loss":
                self.loss = tf.losses.log_loss(self.label, out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, out))
        # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["concat_projection"])
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_%d"%i])
        
            auc_op, auc_value = tf.metrics.auc(self.label, out)
        
            return self.loss, auc_value
