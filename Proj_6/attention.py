import tensorflow as tf
from einops import rearrange
from linear import Linear


class MultiHeadAttention(tf.Module):

    def dot_product_attention(
        query: tf.Tensor, keys: tf.Tensor, value: tf.Tensor, mask=None
    ):
        ###want to preserve the batch dimension for broadcasting
        upper = query @ rearrange(keys, "... h w -> ... w h")
        # Get last dimension of query, dk
        d_k = tf.cast(query.shape[-1], tf.float32)
        scaled_dot_prod = upper / tf.math.sqrt(d_k)
        if mask:
            ##masking: Forcing causality?
            empty = tf.zeros_like(scaled_dot_prod)
            scaled_dot_prod = tf.where(mask, scaled_dot_prod, empty)

        attention = tf.nn.softmax(scaled_dot_prod)
        values = attention @ value

        return values

    def __init__(self, num_inputs, embed_dim, num_heads, dropout_rate=0.2,seed=[0,42]):
        ##initialization
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seed = seed

        ##This division needs to be clean
        if embed_dim % num_heads != 0:
            raise Exception("Embedded dim should be divisible by num_heads")

        self.head_dim = embed_dim // num_heads
        ##For normalization
        # 3 * embed dimension to allow combining qkv into a single matrix
        self.qkv = Linear(num_inputs=num_inputs, num_outputs=3 * embed_dim, bias=False)
        self.out = Linear(num_inputs=embed_dim, num_outputs=num_inputs, bias=False)

    def __call__(self, input: tf.Tensor, mask=None,dropout=False):
        qkv = self.qkv(input)
        ##Output dimensionality of this qkv is Batch size x 3xT
        qkv = tf.reshape(
            qkv, (input.shape[0], input.shape[1], self.num_heads, 3 * self.head_dim)
        )
        qkv = rearrange(
            qkv, "batch len num_head head_dim -> batch num_head len head_dim"
        )
        ##break up into heads,and apply dot product attention: exploiting broadcasting along the batch dimension
        q, k, v = tf.split(qkv, 3, axis=-1)
        values = MultiHeadAttention.dot_product_attention(q, k, v, mask)
        values = rearrange(
            values, "batch num_head len head_dim -> batch len num_head head_dim"
        )
        values = tf.reshape(values, (input.shape[0], input.shape[1], self.embed_dim))

        if dropout:
            current = tf.nn.experimental.stateless_dropout(
                current, self.dropout_rate, seed=self.seed
            )
        out = self.out(values)
        return out
