import math

import tensorflow as tf
from conv2d import Conv2d
from adam import Adam
from mlp import MLP


##Adapted from https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=3KZZ5jU9zCTK
def coordinate_grid(sidelen, dim=2):
    ## Generates a flattened grid of xy coordinates of dimension sidelen
    tensors = tuple(dim * [tf.linspace(-1, 1, sidelen)])
    mgrid = tf.stack(tf.meshgrid(tensors), axis=-1)
    mgrid = tf.reshape(mgrid, dim)
    return mgrid


class Sine_Activation(tf.Module):
    def __init__(self, omega):
        self.omega = omega

    def __call__(self, x):
        return tf.sin(self.omega * x)


class Siren(tf.Module):

    def __init__(
        self,
        num_inputs,
        num_outputs,
        hidden_layer_width,
        hidden_layer_dim,
        initial_omega=30,
        hidden_omega=30.0,
        seed=42,
    ):
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(seed)
        activation = Sine_Activation(hidden_omega)
        ## Modified Linear to accept optional initialization

    def __call__(self, x):
        pass
        


if __name__ == "__main__":
    import os
    import numpy as np
    from matplotlib import pyplot
    from tqdm import trange

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)


    BATCH_SIZE = 128
    NUM_ITERS = 40
    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    print("Making Model")

    # model = Siren(
    #     input_dims=32,
    #     output_dim=100,
    # )
    model = ''

    optimizer = Adam(learning_rate=0.001)

    # Converting batch_size to epochs
    epochs = 0
    total_epochs = BATCH_SIZE * NUM_ITERS / size

    print("Running for %0.4f epochs" % total_epochs)
    n_min = 0.1
    n_max = 2

    bar = trange(NUM_ITERS)
    for i in bar:
        batch_indices = np_rng.integers(low=0, high=size, size=BATCH_SIZE).T
        with tf.GradientTape() as tape:
            # Cross Entropy Loss Function
            loss = tf.keras.losses.categorical_crossentropy(batch_labels, predicted)
            loss = tf.math.reduce_mean(loss)

        epochs += BATCH_SIZE / size
        # Cosine annealing
        n_t = n_min + (n_max - n_min) * (
            1 + tf.math.cos(epochs * math.pi / total_epochs)
        )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if i % 10 == 0:
            bar.set_description(
                f"epoch {epochs:0.4f}; Loss => {loss.numpy():0.4f}:"
            )
            bar.refresh()

    print(
        "On Test Card F, achieved accuracy of %0.1f%%", 100)