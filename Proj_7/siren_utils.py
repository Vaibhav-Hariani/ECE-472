import tensorflow as tf

def coordinate_grid(xlen, ylen, dim=2):
    ## Generates a flattened grid of xy coordinates of dimension xlen, ylen
    tensors = (tf.linspace(-1, 1, xlen), tf.linspace(-1,1,ylen))
    ##TODO: Debug if the stack & reshape are necessary 
    mgrid = tf.meshgrid(tensors[0],tensors[1])
    mgrid = tf.stack(mgrid, axis=-1)
    mgrid = tf.reshape(mgrid, [-1,dim])
    return tf.cast(mgrid,tf.float32)
