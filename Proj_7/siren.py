import math

import tensorflow as tf
from linear import Linear
from PIL import Image
import os


##Adapted from https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=3KZZ5jU9zCTK
def coordinate_grid(xlen, ylen, dim=2):
    ## Generates a flattened grid of xy coordinates of dimension xlen, ylen
    tensors = (tf.linspace(-1, 1, xlen), tf.linspace(-1, 1, ylen))
    ##TODO: Debug if the stack & reshape are necessary
    mgrid = tf.meshgrid(tensors[0], tensors[1])
    mgrid = tf.stack(mgrid, axis=-1)
    mgrid = tf.reshape(mgrid, [-1, dim])
    return tf.cast(mgrid, tf.float32)


class Sine_Activation(tf.Module):
    def __init__(self, omega):
        self.omega = omega

    def __call__(self, x):
        return tf.sin(self.omega * x)


class Siren(tf.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_width,
        hidden_layer_dim,
        initial_omega=30.0,
        hidden_omega=30.0,
        seed=42,
    ):
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(seed)
        self.i_omega = initial_omega
        self.hidden_omega = hidden_omega
        ## Modified Linear to accept optional initialization
        ##***My implementation does not support single layer models
        initial = rng.uniform(
            [input_dim, hidden_layer_dim], -1 / input_dim, 1 / input_dim
        )
        self.initial_layer = Linear(input_dim, hidden_layer_dim, initial=initial)
        self.layers = []
        for x in range(hidden_layer_width):
            r = tf.math.sqrt(6 / hidden_layer_dim) / hidden_omega
            initial = rng.uniform([hidden_layer_dim, hidden_layer_dim], -1 * r, r)
            self.layers.append(
                Linear(hidden_layer_dim, hidden_layer_dim, initial=initial)
            )

        output_range = tf.math.sqrt(6 / hidden_layer_dim) / hidden_omega
        output_initial = rng.uniform(
            [hidden_layer_dim, output_dim], -1 * output_range, output_range
        )
        self.layers.append(Linear(hidden_layer_dim, output_dim, initial=output_initial))

    def __call__(self, x):
        intermediate = tf.math.sin(self.i_omega * self.initial_layer(x))
        for layer in self.layers[:-1]:
            intermediate = tf.math.sin(self.hidden_omega * layer(intermediate))
        return self.layers[-1](intermediate)


def get_image_tensor(image_path, xlen, ylen):
    img = Image.open(image_path)
    array = np.asarray(img)
    image = tf.image.resize(array, (xlen, ylen))
    ##Image is now bound between 0 & 2
    image = image / 128.0
    ##Image is now bound between -1 * 1
    image = image - 1
    return image


def render_img(image, path, label=""):
    # path = os.path.join("testing_imgs", path)
    ##Testing to make sure images are generated properly
    plt.imshow(image)
    plt.ylabel(label)
    plt.savefig(path)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import trange
    from adam import Adam

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    ylen = 365
    xlen = 273

    image = get_image_tensor("Proj_7/Test_Card_F.jpg", xlen, ylen)
    model = Siren(input_dim=2, output_dim=3, hidden_layer_dim=100, hidden_layer_width=3)

    # optimizer = Adam(size=len(model.trainable_variables), step_size=0.001)
    optimizer = tf.optimizers.AdamW(learning_rate=0.001)

    # Converting batch_size to epochs
    NUM_ITERS = 500

    epochs = 0
    total_epochs = NUM_ITERS

    print("Running for %0.1f epochs" % total_epochs)
    n_min = 0.1
    n_max = 2

    bar = trange(NUM_ITERS)

    grid = coordinate_grid(xlen, ylen)
    truth = image
    render_img((truth + 1) / 2, "truth.png")
    truth = tf.reshape(truth, [-1, 3])
    for i in bar:
        with tf.GradientTape() as tape:
            # Cross Entropy Loss Function
            predicted = model(grid)
            # predicted = tf.reshape(predicted, [xlen,ylen,3])
            loss = tf.math.reduce_mean((predicted - truth) ** 2)

        epochs += 1
        # Cosine annealing
        n_t = n_min + (n_max - n_min) * (
            1 + tf.math.cos(epochs * math.pi / total_epochs)
        )
        grads = tape.gradient(loss, model.trainable_variables)
        # optimizer.train(grads=grads, vars=model.trainable_variables, adamW=True, decay_scale=n_t)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if i % 3 == 0:
            bar.set_description(f"Step {i}; Loss => {loss.numpy():0.4f}:")
            bar.refresh()

    output_img = model(grid)

    flattened = tf.reshape(output_img, [xlen, ylen, 3])
    ##Convert this image to a range from 0 - 2 & then 0 -1
    flattened = (flattened + 1) / 2
    render_img(flattened, "model_output.png")

    # loss = tf.math.reduce_mean((flattened - truth)**2)
    print("On Test Card F, achieved accuracy of %0.1f", loss.numpy())
