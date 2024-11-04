from attention import MultiHeadAttention
import tensorflow as tf
from resnet_linear import ResidualBlock
import numpy as np

class Pre_Processor(tf.Module):
    ##This takes in an input string, tokenizes it
    ##Places them in an embedding space,
    ##And then handles positional encoding.
    def __init__(self):
        pass

    def __call__(self, context):
        ##Given context: Break it up into tokens     
        tokens = str.split(context)
        tokens = np.array(tokens)

        embedded_tokens = None
        ##Assuming that tokens get properly parsed here

        Tokens * positional_embeddings = 

class TransformerBlock(tf.Module):
    ##Mask necessary for first block in decoder
    def __init__(self, input_dim, num_heads,dropout_rate=0.2, mask=None, seed = [0,42]):
        self.mask = mask
        self.attn = MultiHeadAttention(input_dim, input_dim, num_heads,dropout_rate=dropout_rate,seed=seed)
        self.ff = ResidualBlock(input_dim=input_dim,output_dim=input_dim,dropout_rate=dropout_rate,seed=seed)

    def call(self, x,dropout=False):
        attention_out = x + self.attn(x,mask=self.mask,dropout=dropout)
        ff = self.ff(attention_out,dropout=dropout)