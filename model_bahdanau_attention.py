#-*- coding: utf-8 -*-

import tensorflow as tf

    
'''
desc : apply bahdanau attention to target vector with given condition

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
def bahdanau_attention( batch_size, target, condition, target_encoder_length, target_dim, condition_dim ) :

    # tile condition
    # [batch, dim] -> [bathc, target_encoder_length, dim]
    batch_embed_given = condition
    batch_seq_embed_given = tf.tile( batch_embed_given, [1,  target_encoder_length] )
    batch_seq_embed_given = tf.reshape( batch_seq_embed_given, [batch_size, target_encoder_length, condition_dim] )


    # concat [ target, condition ] 
    # [ batch, target_encoder_length, (target_dim, condition_dim) ] 
    batch_concat = tf.concat( [target, batch_seq_embed_given], axis=2 )
    
    # make batch
    # --> [ batch x target_encoder_length, (target_dim, condition_dim) ] 
    batch_concat = tf.reshape( batch_concat, [batch_size * target_encoder_length, (target_dim+condition_dim)] )
    
    
    initializers = tf.contrib.layers.xavier_initializer(
                                                            uniform=True,
                                                            seed=None,
                                                            dtype=tf.float32
                                                            )
    
    outputs = tf.contrib.layers.fully_connected( 
                        inputs = batch_concat,
                        num_outputs = 1,
                        activation_fn = tf.nn.tanh,
                        normalizer_fn=None,
                        normalizer_params=None,
                        weights_initializer=initializers,
                        weights_regularizer=None,
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        trainable=True
                        )
    
    
    dot = tf.reshape( outputs, [batch_size, target_encoder_length] )
    
    
    # pad 부분을 작은 값으로 대체 --> 그래야 softmax 후 0으로 떨어짐
    pad_position = tf.equal(tf.reshape(dot, [batch_size, target_encoder_length]), 0.0)
    tmp = tf.to_float(pad_position) * -1e9
    tmp = tf.expand_dims(tmp, 2)
    base = tf.ones( [batch_size, target_encoder_length, 1] ) * tmp
    
    """
    norm_dot = tf.nn.softmax( dot+base, dim=1 )
   
    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply( batch_seq_embed_target, norm_dot )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )
        
    """
    
    return dot, base
    