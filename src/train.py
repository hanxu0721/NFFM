#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:10:53 2019

@author: hanxu8
"""
import os 
import numpy as np
import tensorflow as tf
import nffm_data_parser 
from nffm import NFM
import config

# config
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("factor_size", 256, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_integer("field_size", 15, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_integer('max_epoch', 20, ' max train epochs')
flags.DEFINE_integer("batch_size", 200, "batch size for sgd")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "checkpoint dirctory")
flags.DEFINE_string("optimizer", "adam", "optimization algorithm")
flags.DEFINE_string("loss", "log_loss", "loss function")
flags.DEFINE_string("deep_layers", "256-128", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_string("dropout_out", "0.9-0.9-0.9", "Dropout keep probability (default: 0.5)")
flags.DEFINE_integer("batch_norm", 0, "batch_norm")
flags.DEFINE_float('batch_norm_decay', 0.995, 'batch_norm_decay')
flags.DEFINE_float("l2_reg",    0.001, "l2_reg")


def training(path):
    data_train,y_train = nffm_data_parser.decode(path)
    model = NFM(embedding_size = FLAGS.factor_size, 
                 field_size = FLAGS.field_size, 
                 total_fea_dict = config.total_fea_dict,
                 deep_layers = [int(item) for item in FLAGS.deep_layers.split('-')],
                 loss_type=FLAGS.loss,
                 batch_norm = FLAGS.batch_norm,
                 batch_norm_decay = FLAGS.batch_norm_decay,
                 l2_reg = FLAGS.l2_reg
                 )
    drop_out = [float(item) for item in FLAGS.dropout_out.split('-')]
    print("Optimization algorithm: {}".format(FLAGS.optimizer))
    if FLAGS.optimizer   == "sgd":
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    elif FLAGS.optimizer == "adadelta":
        optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate)
    elif FLAGS.optimizer == "adagrad":
        optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    elif FLAGS.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    else:
        print("Error: unknown optimizer: {}".format(FLAGS.optimizer))
        exit(1)
        
    loss,auc,out = model.loss()
    #_, train_auc  = tf.contrib.metrics.streaming_auc(predict_label, true_label)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op    = optimizer.minimize(loss, global_step = global_step)
    saver = tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        for epoch in range(FLAGS.max_epoch):
            print ("eopch={}".format(epoch))
            nffm_data_parser.shuffle_in_unison_scary(data_train,  y_train)
            total_batch = int(len(y_train) / FLAGS.batch_size)
            print ("total_batch={}".format(total_batch))
            for i in range(total_batch):
                data_batch,y_batch = nffm_data_parser.get_batch(data_train, y_train, FLAGS.batch_size, i)
                sparse_dict = nffm_data_parser.get_input2(data_batch,config.field)
                feed_dict = {}
                for i in config.field:
                    feed_dict[model.features[i]] = sparse_dict[i]
                    #print sparse_dict[i]
                feed_dict[model.label] = y_batch
                feed_dict[model.dropout_keep_deep] = drop_out
                feed_dict[model.train_phase] = True
                _, loss_value, auc_value, out_value, step = sess.run([train_op, loss, auc, out, global_step], feed_dict=feed_dict)
                print ("step={} ,loss={:.5f}, auc={:.5f}, out={}".format(step,loss_value,auc_value, out_value))
        saver.save(sess,'/data3/ads_dm/hanxu8/nFFM/model/nffm.ckpt',global_step=step)

path = '/data3/ads_dm/hanxu8/nFFM/deal_data/fea2index'
filenames = tf.gfile.ListDirectory(path)
filenames = [x for x in filenames]
filenames = [os.path.join(path, x) for x in filenames]
filenames = [x for x in filenames if tf.gfile.Exists(x)]
ins_list = []
for item in filenames:
    ins_list.append(item)
for ins in ins_list:
    training(ins)
