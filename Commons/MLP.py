import tensorflow as tf
from linear import Linear


# My Little Perceptron
class MLP(tf.Module):
    # A multi layer perceptron implementation:

    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
        dropout_rate=0,
        seed= [42,0]
    ):
        self.seed=seed
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout_rate=dropout_rate

        self.linear_steps = []
        if num_hidden_layers == 0:
            lin_obj = Linear(num_inputs, num_outputs)
            self.linear_steps.append(lin_obj)

        else:
            obj1 = Linear(num_inputs, hidden_layer_width)
            self.linear_steps.append(obj1)
            for x in range(0, num_hidden_layers - 1):
                lin_obj = Linear(hidden_layer_width, hidden_layer_width)
                self.linear_steps.append(lin_obj)
            final_obj = Linear(hidden_layer_width, num_outputs)
            self.linear_steps.append(final_obj)

    def __call__(self, layer_in,dropout=False):
        current = layer_in
        for i in self.linear_steps[:-1]:
            current = i(current)
            current = self.hidden_activation(current)
            if dropout:
                current=tf.nn.experimental.stateless_dropout(current,self.dropout_rate,seed=self.seed)
        current = self.linear_steps[-1](current)
        return self.output_activation(current)
