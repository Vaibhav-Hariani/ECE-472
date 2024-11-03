from Proj_6.attention import MultiHeadAttention
import tensorflow as tf
from groupnorm import GroupNorm
from mlp import MLP


class TransformerBlock(tf.Module):
    def __init__(self, input_dim, num_heads, dim_ff, dropout_rate=0, seed=[42, 0]):
        self.attn = MultiHeadAttention(input_dim, input_dim, num_heads)
        self.add = MLP(num_inputs=input_dim, num_outputs=input_dim)

