from attention import MultiHeadAttention
import tensorflow as tf
from resnet_linear import ResidualBlock

class TransformerBlock(tf.Module):
    ##Mask necessary for first block in decoder
    def __init__(self, input_dim, num_heads,dropout_rate=0.2, mask=None, seed = [0,42]):
        self.mask = mask
        self.attn = MultiHeadAttention(input_dim, input_dim, num_heads,dropout_rate=dropout_rate,seed=seed)
        self.ff = ResidualBlock(input_dim=input_dim,output_dim=input_dim,dropout_rate=dropout_rate,seed=seed)

    def call(self, x,dropout=False):
        attention_out = x + self.attn(x,mask=self.mask,dropout=dropout)
        ff = self.ff(attention_out,dropout=dropout)