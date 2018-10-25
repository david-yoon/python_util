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
def KimCNN(_input, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, dr_CNN=1.0):

    with tf.variable_scope("kimCnn", reuse=False, initializer=tf.contrib.layers.xavier_initializer()):

        embedded_expanded = tf.expand_dims(_input, -1)

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                #b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                #self.h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                h = tf.nn.relu( conv, name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)

        h_pool      = tf.concat(pooled_outputs, 3)        
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, dr_CNN)

        """
        W = tf.get_variable(
            "W_cnn",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_cnn")
        #l2_loss += tf.nn.l2_loss(W)
        #l2_loss += tf.nn.l2_loss(b)
        final_output = tf.tanh( tf.nn.xw_plus_b(h_drop, W, b, name="final_output") )


        #special case reduced to batch
        h_mask   = tf.sequence_mask( self.batch_num_a, Params.MAX_ANSWERS, dtype=tf.float32)
        h_mask   = tf.reshape( h_mask, [self.original_batch_size * Params.MAX_ANSWERS, 1] )
        final_output_masked = tf.multiply( final_output, h_mask )
        """

    return h_drop, num_filters_total