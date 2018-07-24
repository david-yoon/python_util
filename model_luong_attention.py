#-*- coding: utf-8 -*-

import tensorflow as tf

    
'''
desc : apply luong attention to target vector with given condition

input :
   - batch_size             : 
   - target                 : [batch, seq, embed]
   - condition              : [batch, embed] --> last hidden
   - target_encoder_length  : max encoder length
   - hidden                 : should be same btw target and condition, otherwise code should be changed

output : 
   - attented target : weighted sum [batch, embed]
   - norm_dot : attention weight
'''
def luong_attention( batch_size, target, condition, target_encoder_length, hidden_dim ) :

    # same dim [batch, max_seq, embed]
    batch_seq_embed_target = tf.reshape( target, [batch_size, target_encoder_length, hidden_dim] )
    
    batch_embed_given = condition
    batch_seq_embed_given = tf.reshape( batch_embed_given, [batch_size,  hidden_dim, 1] )
    
    # calculate similarity 
    dot = tf.matmul( batch_seq_embed_target,  batch_seq_embed_given )
    
    # pad 부분을 작은 값으로 대체 --> 그래야 softmax 후 0으로 떨어짐
    pad_position = tf.equal(tf.reshape(dot, [batch_size, target_encoder_length]), 0.0)
    tmp = tf.to_float(pad_position) * -1e9
    tmp = tf.expand_dims(tmp, 2)
    base = tf.ones( [batch_size, target_encoder_length, 1] ) * tmp
    
    norm_dot = tf.nn.softmax( dot+base, dim=1 )
   
    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply( batch_seq_embed_target, norm_dot )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )

    return weighted_sum, norm_dot
    
    
    
'''
desc : apply luong attention to target vector with given condition

input :
   - batch_size             : 
   - target                 : [batch, seq, embed]
   - condition              : [batch, embed] --> last hidden
   - target_encoder_length  : max encoder length
   - hidden                 : should be same btw target and condition, otherwise code should be changed

output : 
   - attented target : weighted sum [batch, embed]
   - norm_dot : attention weight
'''
def luong_attention_mul_condition( batch_size, target, condition, target_dim, condition_dim, max_target_encoder_length, attn_dim) :

    weighted_sum = 0
    norm_dot = 0
  
    W_target = tf.Variable(tf.random_uniform([target_dim, attn_dim],
                                                  minval= -0.25,
                                                  maxval= 0.25,
                                                  dtype=tf.float32,
                                                  seed=None),
                           trainable=True,
                           name="attn_W_target")
    
    
    W_condition = tf.Variable(tf.random_uniform([condition_dim,attn_dim],
                                                  minval= -0.25,
                                                  maxval= 0.25,
                                                  dtype=tf.float32,
                                                  seed=None),
                                                     trainable=True,
                                                     name="attn_W_condition")
    
    attn_bias = tf.Variable(tf.zeros([1], dtype=tf.float32),
                                                 trainable=True,
                                                 name="attn_bias")
    
    W_target = tf.reshape( W_target, [1, target_dim, attn_dim])
    W_target = tf.tile( W_target, [batch_size, 1, 1])
    
   
    W_condition = tf.reshape( W_condition, [1, target_dim, attn_dim])
    W_condition = tf.tile( W_condition, [batch_size, 1, 1])
    
    tmp_target = tf.matmul( target, W_target )
    tmp_condition = tf.matmul( condition, W_condition )
    
    dot = tf.multiply(tmp_target, tmp_condition)
    dot = tf.reduce_sum(dot, axis=2) + attn_bias
    
    # pad 부분을 작은 값으로 대체 --> 그래야 softmax 후 0으로 떨어짐
    pad_position = tf.equal(tf.reshape(dot, [batch_size, max_target_encoder_length]), 0.0)
    base = tf.to_float(pad_position) * -1e9
    
    norm_dot = tf.nn.softmax( dot+base, dim=1 )
    norm_dot = tf.reshape( norm_dot, [batch_size, max_target_encoder_length, 1] )
   
    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply( target, norm_dot )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )

    return weighted_sum, norm_dot
