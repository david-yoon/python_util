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
'''
def luong_attention( batch_size, target, condition, target_encoder_length, hidden_dim ) :

    # same dim [batch, max_seq, embed]
    batch_seq_embed_target = tf.reshape( target, [batch_size, target_encoder_length, hidden_dim] )
    
    batch_embed_given = condition
    batch_seq_embed_given = tf.reshape( batch_embed_given, [batch_size,  hidden_dim, 1] )
    
    # calculate similarity 
    dot = tf.matmul( batch_seq_embed_target,  batch_seq_embed_given )
    norm_dot = tf.nn.softmax( dot, dim=1 )
    
    # weighted sum by using similarity (normalized)
    target_mul_norm = tf.multiply( batch_seq_embed_target, norm_dot )
    weighted_sum = tf.reduce_sum( target_mul_norm, axis=1 )

    return weighted_sum


'''
desc : latent topic cluster method

input :
   - batch_size          : 
   - topic                  : # of topics
   - memory_dim       : dim of each topic
   - hidden_dim         : dim of input vector
   - input_encoder     : [batch, dim_encoder]
   - dr_memory_prob  : dropout ratio for memory

output : 
   - final_encoder : LTC applied vector [batch, vector(== concat(original, topic_mem)]
   - final_encoder_dimension : 
'''
def sy_ltc( batch_size, topic_size, memory_dim, hidden_dim, input_encoder, dr_memory_prob=1.0 ):
    print '[launch] s.y. latent topic cluster method'

    with tf.name_scope('memory_network_v1') as scope:

        # memory space for latent topic
        memory = tf.get_variable( "latent_topic_memory", 
                                      shape=[topic_size, memory_dim],
                                      initializer=tf.orthogonal_initializer()
                                     )

        memory_W = tf.Variable(tf.random_uniform( [hidden_dim, memory_dim],
                                                      minval= -0.25,
                                                      maxval= 0.25,
                                                      dtype=tf.float32,
                                                      seed=None),
                                    name="memory_projection_W")

        memory_W = tf.nn.dropout( memory_W, keep_prob=dr_memory_prob )
        memory_bias = tf.Variable(tf.zeros([1], dtype=tf.float32), name="memory_projection_bias")

        topic_sim_project = tf.matmul( input_encoder, memory_W ) + memory_bias

        # context 와 topic 의 similairty 계산
        topic_sim = tf.matmul( topic_sim_project, memory, transpose_b=True )
        #topic_sim_sigmoid = tf.sigmoid( topic_sim )

        # normalize
        topic_sim_sigmoid_softmax = tf.nn.softmax( logits=topic_sim, dim=-1)
        #self.topic_sim_sigmoid_softmax = tf.nn.softmax( logits=topic_sim_sigmoid, dim=-1)

        # memory_context 를 계산  memory 를 topic_sim_norm 으로 weighted sum 수행
        # batch_size = 1 인 경우를 위해서 shape 을 맞추어줌
        # batch_size > 1 인 경우는 원래 형태와 변화가 없음
        shaped_input = tf.reshape( topic_sim_sigmoid_softmax, [batch_size, topic_size])

        topic_sim_mul_memory = tf.scan( lambda a, x : tf.multiply( tf.transpose(memory), x ), shaped_input, initializer=tf.transpose(memory) )
        tmpT = tf.reduce_sum(topic_sim_mul_memory, axis=-1, keep_dims=True)
        tmpT2 = tf.transpose(tmpT, [0, 2, 1])

        rsum = tf.reshape( tmpT2, [batch_size, memory_dim])

        # final context 
        final_encoder  = tf.concat( [input_encoder, rsum], axis=-1 )
        #self.final_encoderR = tf.concat( [self.final_encoderR, rsum], axis=-1 )

        final_encoder_dimension  = hidden_dim + memory_dim   # concat 으로 늘어났음
        #self.final_encoderR_dimension = hidden_dim
        #self.final_encoderR_dimension = hidden_dim + memory_dim   # encoder 에서 구한 latent topic 값을 같이 사용
                
        return final_encoder, final_encoder_dimension