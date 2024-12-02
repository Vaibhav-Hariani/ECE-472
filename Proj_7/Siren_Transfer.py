import os

import tensorflow as tf
from linear import Linear
from resnet_conv import ResNet


# Adapted from https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb#scrollTo=3KZZ5jU9zCTK
def coordinate_grid(xlen, ylen, dim=2):
    # Generates a flattened grid of xy coordinates of dimension xlen, ylen
    tensors = (tf.linspace(-1, 1, xlen), tf.linspace(-1, 1, ylen))
    # TODO: Debug if the stack & reshape are necessary
    mgrid = tf.meshgrid(tensors[0], tensors[1])
    mgrid = tf.stack(mgrid, axis=-1)
    mgrid = tf.reshape(mgrid, [-1, dim])
    return tf.cast(mgrid, tf.float32)


# ADAin with a SIREN
class Gen_Siren(tf.Module):

    def __init__(
        self,
        siren_in,
        siren_out,
        hidden_layer_width,
        hidden_layer_dim,
        initial_omega=30.0,
        hidden_omega=30.0,
        xlen=32,
        seed=42,
    ):
        # This architecture only supports square images (to boot)
        # Parameters for siren
        # Size of every linear layer is input_dim * output_dim + output_dim

        # Dimensionality of each of the blocks on the outset of the resnet

        rng = tf.random.get_global_generator()
        rng.reset_from_seed(seed)

        ## 2 parameters for every layer: gamma & beta
        out_params = hidden_layer_width * 2 + 4

        self.Generator = ResNet(
            xlen,
            output_dim=out_params,
            conv_dims=3,
            hidden_lin_width=32,
            num_lin_layers=3,
            conv_activation=tf.nn.relu,
            lin_activation=tf.nn.leaky_relu,
            lin_output_activation=tf.nn.relu,
        )

        self.i_omega = initial_omega
        self.hidden_omega = hidden_omega

        ##Siren initialization here
        initial = rng.uniform([siren_in, hidden_layer_dim], -1 / siren_in, 1 / siren_in)
        self.initial_layer = Linear(siren_in, hidden_layer_dim, initial=initial)
        self.layers = []
        for x in range(hidden_layer_width):
            r = tf.math.sqrt(6 / hidden_layer_dim) / hidden_omega
            initial = rng.uniform([hidden_layer_dim, hidden_layer_dim], -1 * r, r)
            self.layers.append(
                Linear(hidden_layer_dim, hidden_layer_dim, initial=initial)
            )
        output_range = tf.math.sqrt(6 / hidden_layer_dim) / hidden_omega
        output_initial = rng.uniform(
            [hidden_layer_dim, siren_out], -1 * output_range, output_range
        )
        self.layers.append(Linear(hidden_layer_dim, siren_out, initial=output_initial))

    def __call__(self, siren_inputs, resnet_inputs):
        resnet_out = self.Generator(resnet_inputs)
        i = 0
        intermediate = tf.math.sin(self.i_omega * self.initial_layer(siren_inputs))
        for layer in self.layers[:-1]:
            current = layer(intermediate)
            current = current * resnet_out[:, i] + resnet_out[:, i + 1]
            i += 2
            intermediate = tf.math.sin(current)

        # Final layer is not sined
        # intermediate = self.layers[-1](intermediate)
        ##bounding it from 0 to 1

        # return self.layers[-1](intermediate)

        # return tf.nn.sigmoid(self.layers[-1](intermediate))
        return resnet_out[:, -2] * self.layers[-1](intermediate) + resnet_out[:, -1]


def get_image_tensor(img, xlen, ylen):
    # img = Image.open(image_path)
    # array = np.asarray(img)
    image = tf.image.resize(img, (xlen, ylen))
    # Image is now bound between 0 & 2
    image = image / 128.0
    # Image is now bound between -1 * 1
    image = image - 1
    # image = image / 255.0
    return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from adam import Adam
    from CIFAR_UTILS import render_img, unpickle
    from tqdm import trange

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    ylen = 32
    xlen = 32

    # Getting training data from local CIFAR
    CIFAR_LOC = "CIFAR"
    CIFAR_FOLDER = "cifar-100-python"
    IMG_DIMS = (-1, 3, 32, 32)
    TRAIN_SET = "train"

    path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, TRAIN_SET)
    raw_dict = unpickle(path)

    images = np.reshape(raw_dict[b"data"], IMG_DIMS)
    train_size = 1
    size = train_size
    images = np.reshape(raw_dict[b"data"], IMG_DIMS)
    images = np.transpose(images, (0, 2, 3, 1))

    model = Gen_Siren(
        siren_in=2, siren_out=3, hidden_layer_dim=255, hidden_layer_width=5
    )

    # optimizer = Adam(size=len(model.trainable_variables), step_size=0.001)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)

    # Train on one image at a time
    BATCH_SIZE = 1
    NUM_ITERS = 1000

    epochs = 0
    total_epochs = NUM_ITERS

    print("Running for %0.1f epochs" % total_epochs)
    n_min = 0.1
    n_max = 2

    bar = trange(NUM_ITERS)
    grid = coordinate_grid(xlen, ylen)
    # truth = image
    # render_img((truth + 1) / 2, "truth.png")
    # truth = tf.reshape(truth, [-1, 3])
    for i in bar:
        image_id = np_rng.integers(low=0, high=size, size=BATCH_SIZE).T
        image = get_image_tensor(images[image_id], xlen, ylen)
        with tf.GradientTape() as tape:
            # Cross Entropy Loss Function
            predicted = model(grid, image)
            predicted = tf.reshape(predicted, [1, xlen, ylen, 3])
            # predicted = (predicted + 1 / 2)
            # image = (image + 1 / 2)
            # loss = tf.keras.losses.categorical_crossentropy(image, predicted)
            loss = (predicted - image) ** 2
            loss = tf.math.reduce_mean(loss)

        # epochs += 1
        # # Cosine annealing
        # n_t = n_min + (n_max - n_min) * (
        #     1 + tf.math.cos(epochs * math.pi / total_epochs)
        # )
        grads = tape.gradient(loss, model.trainable_variables)
        # optimizer.train(grads=grads, vars=model.trainable_variables, adamW=True, decay_scale=n_t)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if i % 3 == 0:
            bar.set_description(f"Step {i}; Loss => {loss.numpy():0.4f}:")
            bar.refresh()

    render_img(images[0], "final_truth.png")

    final_img = get_image_tensor(images[[0]], xlen, ylen)

    output_img = model(grid, final_img)
    model_out = tf.reshape(output_img, [xlen, ylen, 3])
    # Convert this image to a range from 0 - 2 & then 0 -1
    model_out = (model_out + 1) / 2
    render_img(model_out, "model_output_general.png")

    loss = (final_img - model_out) ** 2
    loss = tf.math.reduce_mean(loss)
    # loss = tf.math.reduce_mean((flattened - truth)**2)
    print(f"On New sample, achieved accuracy of {loss.numpy():0.4f}")
