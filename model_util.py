#-*- coding: utf-8 -*-

import tensorflow as tf


'''
desc : apply luong attention to target vector with given condition

input :
   - target         : [batch, seq, embed]
   - condition    : [batch, embed] --> last hidden
   - batch_size :  
   - target_seq : 
   - hidden       : should be same btw target and condition, otherwise code should be changed

output : 
   - attented target : weighted sum [batch, embed]
    '''
def luong_attention( target, condition, batch_size, target_seq_length, hidden_dim ) :

    batch_seq_embed_target = tf.reshape( target, [batch_size, target_seq_length, hidden_dim] )
    batch_embed_given = condition


    batch_seq_embed_given = tf.reshape( batch_embed_given, [batch_size,  hidden_dim, 1] )
    dot = tf.matmul( batch_seq_embed_target,  batch_seq_embed_given )
    norm_dot = tf.nn.softmax( dot, dim=1 )
    target_mul_norm = tf.multiply( batch_seq_embed_target, norm_dot )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )

    return weighted_sum        