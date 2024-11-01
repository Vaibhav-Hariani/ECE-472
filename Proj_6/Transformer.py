import tensorflow as tf


class MultiHeadAttention(tf.Module):

    def dot_product_attention(query:tf.Tensor,keys:tf.Tensor,value:tf.Tensor, mask=None):
        upper = query @ keys.T
        #Get second dimension of query, dk
        scaled_dot_prod = upper / tf.math.sqrt(query.shape(1))
        if mask:
            ##masking: Not sure if this will be necessary, but can't hurt?
            empty = tf.zeros_like(scaled_dot_prod)
            scaled_dot_prod = tf.where(mask,scaled_dot_prod,empty) 

        return tf.nn.softmax(scaled_dot_prod) @ value

    def __init__(self,input_dim,embed_dim, num_heads):
        rng = tf.random.get_global_generator()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        ##This division needs to be clean
        self.head_dim = embed_dim // num_heads

        ##For normalization
        # 3 * embed dimension to allow combining qkv into a single matrix
        qkv_stddev = 2 / (input_dim*3*embed_dim)
        self.qkv_mats = tf.Variable(
            rng.normal(stddev=qkv_stddev, shape=[input_dim,3*embed_dim]),
            trainable=True
        )
        out_stddev = 2 / (input_dim*embed_dim)
        self.out_mat = tf.Variable(
            rng.normal(stddev=out_stddev, shape=[embed_dim,embed_dim]),
            trainable=True
        )


    

    def __call__():
        pass
        
#Generates Embeddings & Positional Encodings 
#Given an input
class Encoder(tf.Module):

    def tokenizer():
        pass


    def __call__():
        pass


class Transformer(tf.Module):
    def __call__():
        pass




