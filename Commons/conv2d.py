import tensorflow as tf


class Conv2d(tf.Module):
    def __init__(self, dim, in_channel=1, out_channel=1, dropout_rate=0, seed=[42, 0]):

        rng = tf.random.get_global_generator()
        # Kaiming Initialization, from the paper we read
        node_stddev = 2 / (dim * dim * in_channel * out_channel)
        self.kernel = tf.Variable(
            rng.normal(stddev=node_stddev, shape=[dim, dim, in_channel, out_channel]),
            trainable=True,
        )
        self.dropout_rate = dropout_rate
        self.seed = seed

    def __call__(self, input, dropout=False):
        conv = tf.nn.conv2d(input, self.kernel, strides=1, padding="SAME")
        if dropout:
            return tf.nn.experimental.stateless_dropout(
                conv, self.dropout_rate, seed=self.seed
            )
        return conv
