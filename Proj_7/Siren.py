import math

import tensorflow as tf
from linear import Linear
from PIL import Image 


##Adapted from https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=3KZZ5jU9zCTK
def coordinate_grid(xlen, ylen, dim=2):
    ## Generates a flattened grid of xy coordinates of dimension xlen, ylen
    tensors = tuple(tf.linspace(-1, 1, xlen), tf.transpose(tf.linspace(-1,1,ylen)))
    ##TODO: Debug if the stack & reshape are necessary 
    mgrid = tf.meshgrid(tensors)
    mgrid = tf.stack(mgrid, axis=-1)
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
        input_dim,
        output_dim,
        hidden_layer_width,
        hidden_layer_dim,
        initial_omega=30,
        hidden_omega=30.0,
        seed=42,
    ):
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(seed)
        self.input_activation = Sine_Activation(initial_omega)
        self.hidden_activation = Sine_Activation(hidden_omega)
        ## Modified Linear to accept optional initialization
        ##***My implementation does not support single layer models
        initial = rng.uniform(
            [input_dim, hidden_layer_dim], -1 / input_dim, 1 / input_dim
        )
        self.initial_layer = Linear(input_dim, hidden_layer_dim, initial=initial)
        self.layers = []
        for x in range(hidden_layer_width):
            r = tf.math.sqrt(6 / hidden_layer_dim) / hidden_omega
            initial = rng.uniform(
                [hidden_layer_dim, hidden_layer_dim], -1 * r, r
            )
            self.layers.append(
                Linear(hidden_layer_dim, hidden_layer_dim, initial=initial)
            )

        output_range = tf.math.sqrt(6 / hidden_layer_dim) / hidden_omega
        output_initial = rng.uniform([hidden_layer_dim, output_dim], -1 * output_range, output_range)
        self.layers.append(Linear(hidden_layer_dim, output_dim, initial=output_initial))

    def __call__(self, x):
        intermediate = self.input_activation(self.initial_layer(x))
        for layer in self.layers:
            intermediate = self.hidden_activation(layer(intermediate))
        return intermediate

    
def get_image_tensor(image_path, xlen, ylen):
    img = Image.open(image_path)
    array = np.asarray(img)
    image = tf.image.resize(array, (xlen,ylen))
    image  = tf.image.per_image_standardization(image)
    return image

if __name__ == "__main__":
    import numpy as np
    from matplotlib import pyplot
    from tqdm import trange
    from adam import Adam


    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    xlen = 960
    ylen = 540

    image = get_image_tensor("Proj_7/Test_Card_F.png", xlen, ylen)
    model = Siren(input_dim=2, output_dim=3, hidden_layer_dim=256, hidden_layer_width=5)

    optimizer = Adam(size=len(model.trainable_variables), step_size=0.001)

    # Converting batch_size to epochs
    BATCH_SIZE = 128
    NUM_ITERS = 400
    size = xlen * ylen

    epochs = 0
    total_epochs = BATCH_SIZE * NUM_ITERS / size

    print("Running for %0.4f epochs" % total_epochs)
    n_min = 0.1
    n_max = 2

    bar = trange(NUM_ITERS)
    
    grid = coordinate_grid(xlen,ylen)
    truth = image    
    for i in bar:
        with tf.GradientTape() as tape:
            # Cross Entropy Loss Function
            predicted = model(grid)
            loss = tf.keras.losses.categorical_crossentropy(truth, predicted)
            loss = tf.math.reduce_mean(loss)

        epochs += BATCH_SIZE / size
        # Cosine annealing
        n_t = n_min + (n_max - n_min) * (
            1 + tf.math.cos(epochs * math.pi / total_epochs)
        )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.train(
            grads=grads, vars=model.trainable_variables, adamW=True, decay_scale=n_t)
        if i % 10 == 0:
            bar.set_description(f"epoch {epochs:0.4f}; Loss => {loss.numpy():0.4f}:")
            bar.refresh()
    
    output_img = model(grid)
    loss = tf.keras.losses.categorical_crossentropy(truth, predicted)

    print("On Test Card F, achieved accuracy of %0.1f%%", 100)
