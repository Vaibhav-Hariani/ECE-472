import tensorflow as tf
from conv2d import Conv2d
from groupnorm import GroupNorm
from linear import Linear

class ResidualBlock(tf.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        groups=1,
        num_layers=2,
        activation=tf.nn.relu,
        dropout_rate=0.2,
        seed=[42, 0],
    ):
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.num_layers = num_layers
        self.input_layer = Linear(num_inputs=input_dim,num_outputs=output_dim)
        self.GNs = []
        self.layers = []
        # Start from second element
        for layer in range(num_layers):
            self.GNs.append(GroupNorm(groups=groups, channels=output_dim))
            self.layers.append(
                Linear(num_inputs=output_dim,num_outputs=output_dim)
            )

        self.activation = activation

    def __call__(self, input, dropout=False):
        # Using this to prevent soft copying
        input = self.input_layer(input)
        intermediate = input
        for i in range(self.num_layers):
            if(dropout):
                intermediate = tf.nn.experimental.stateless_dropout(
                intermediate, self.dropout_rate, seed=self.seed
                )

            intermediate = self.GNs[i](intermediate)
            intermediate = self.activation(intermediate)
            intermediate = self.layers[i](intermediate)
        return input + intermediate
