import tensorflow as tf


##Created by Professor Curro
class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True, initial=None):
        rng = tf.random.get_global_generator()

        if initial is None:
            stddev = 2 / (num_inputs + num_outputs)
            initial = rng.normal(shape=[num_inputs, num_outputs], stddev=stddev)        

        self.w = tf.Variable(
            initial,
            trainable=True,
            name="Linear/w",
        )

        self.bias = bias
        if self.bias:
            self.b = tf.Variable(
                tf.zeros(shape=[1, num_outputs]),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z
