import tensorflow as tf
from conv2d import Conv2d
from groupnorm import GroupNorm
from mlp import MLP


class ResidualBlock(tf.Module):
    def __init__(
        self,
        dim,
        groups,
        activation=tf.nn.relu,
        dropout_rate=0.2,
        in_channels=3,
        out_channels=3,
        num_layers=2,
    ):
        self.num_layers = num_layers
        self.input_conv = Conv2d(
            dim=1, in_channel=in_channels, out_channel=out_channels
        )
        self.GNs = []
        self.convs = []
        # Start from second element
        for layer in range(num_layers):
            self.GNs.append(GroupNorm(groups=groups, channels=out_channels))
            self.convs.append(
                Conv2d(
                    dim=dim,
                    in_channel=out_channels,
                    out_channel=out_channels,
                    dropout_rate=dropout_rate,
                )
            )

        self.activation = activation

    def __call__(self, input, dropout=False):
        # Using this to prevent soft copying
        input = self.input_conv(input)
        intermediate = input
        for i in range(self.num_layers):
            intermediate = self.GNs[i](intermediate)
            intermediate = self.activation(intermediate)
            intermediate = self.convs[i](intermediate, dropout)

        return input + intermediate


class ResNet(tf.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        pool_dims=1,
        conv_dims=5,
        num_lin_layers=5,
        group_sizes=[3, 3, 3, 3, 3],
        hidden_lin_width=125,
        conv_activation=tf.identity,
        lin_activation=tf.identity,
        lin_output_activation=tf.identity,
        dropout_rate=0.1,
        channel_scales=[3, 3, 3, 3, 3],
    ):
        # Channel scales represents how the channels shift throughout
        # This encodes the input channel, all the way up to the output channel
        self.input_dims = input_dims
        self.res_layers = []

        for res_layer in range(0, len(channel_scales) - 1):
            self.res_layers.append(
                ResidualBlock(
                    dim=conv_dims,
                    in_channels=channel_scales[res_layer],
                    out_channels=channel_scales[res_layer + 1],
                    num_layers=2,
                    groups=group_sizes[res_layer + 1],
                    activation=conv_activation,
                    dropout_rate=dropout_rate,
                )
            )
        self.pool_dims = pool_dims
        self.padding = "SAME"

        perceptron_dims = (input_dims * input_dims * channel_scales[-1]) // (
            pool_dims * pool_dims
        )
        self.output_perceptron = MLP(
            num_inputs=perceptron_dims,
            num_outputs=output_dim,
            num_hidden_layers=num_lin_layers,
            hidden_layer_width=hidden_lin_width,
            hidden_activation=lin_activation,
            output_activation=lin_output_activation,
            dropout_rate=dropout_rate,
        )

    def __call__(self, input, dropout=False):
        current = input
        for layer in self.res_layers:
            current = layer(current, dropout)
        # Define the max pooling operation

        pooled_out = tf.nn.avg_pool2d(
            current,
            ksize=self.pool_dims,
            strides=self.pool_dims,
            padding=self.padding,
        )
        perceptron_in = tf.reshape(pooled_out, (pooled_out.shape[0], -1))
        # Flattens for perceptron
        return self.output_perceptron(perceptron_in, dropout)
