import tensorflow as tf
from conv2d import Conv2d
from groupnorm import GroupNorm


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
