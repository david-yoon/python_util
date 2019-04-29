#-*- coding: utf-8 -*-

import tensorflow as tf

"""
Kim Yoon's CNN
sequence_length:
num_classes:
embedding_size:
filter_sizes:
num_filters:
"""
def KimCNN(_input, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, dr_CNN=1.0, scope="KimCNN", reuse=False):

    with tf.variable_scope(name_or_scope=scope, reuse=reuse, initializer=tf.contrib.layers.xavier_initializer()):

        embedded_expanded = tf.expand_dims(_input, -1)

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name="W-"+ str(filter_size) +scope)
                b = tf.get_variable(initializer=tf.zeros(shape=[num_filters]), name="bias-"+ str(filter_size) +scope)
                conv = tf.nn.conv2d(
                    embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"+scope)
                # Apply nonlinearity
                #self.h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h = tf.nn.relu( conv + b, name="relu"+scope)
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool"+scope)
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)

        h_pool      = tf.concat(pooled_outputs, 3)        
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"+scope):
            h_drop = tf.nn.dropout(h_pool_flat, dr_CNN)

    return h_drop, num_filters_total