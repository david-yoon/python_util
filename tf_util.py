import tensorflow as tf

"""
x1, x2: input tensor 
return: cosine similarity among two tensor
"""
def batch_cosine_sim(x1, x2, axis=-1, name='batch_cosine_loss'):
    with tf.name_scope(name):
        x1_val = tf.sqrt(tf.reduce_sum(tf.multiply(x1, x1), axis=-1))
        x2_val = tf.sqrt(tf.reduce_sum(tf.multiply(x2, x2), axis=-1))
        denom = tf.multiply(x1_val,x2_val)
        num = tf.reduce_sum(tf.multiply(x1, x2), axis=axis)
        return tf.div(num,denom)