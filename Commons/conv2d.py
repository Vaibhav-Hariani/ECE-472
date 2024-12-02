import tensorflow as tf
from mlp import MLP

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

class CNN(tf.Module):

    def __init__(
        self,
        input_dims,
        output_dim,
        conv_dims=3,
        num_convs=4,
        num_lin_layers=5,
        hidden_lin_width=125,
        lin_activation=tf.identity,
        lin_output_activation=tf.identity,
        dropout_rate=0.1,
    ):
        self.input_dims = input_dims
        self.convs = []
        for i in range(0, num_convs):
            self.convs.append(Conv2d(conv_dims, dropout_rate=dropout_rate))

        self.perceptron = MLP(
            num_inputs=input_dims**2,
            num_outputs=output_dim,
            num_hidden_layers=num_lin_layers,
            hidden_layer_width=hidden_lin_width,
            hidden_activation=lin_activation,
            output_activation=lin_output_activation,
            dropout_rate=dropout_rate,
        )

    def __call__(self, input, dropout=False):
        current = input
        for conv in self.convs:
            current = conv(current, dropout)

        current = tf.reshape(current, (current.shape[0], -1))
        return self.perceptron(current, dropout)
