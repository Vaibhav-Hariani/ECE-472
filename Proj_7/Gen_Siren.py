import math

import tensorflow as tf
from PIL import Image
import os
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

        ##Dimensionality of each of the blocks on the outset of the resnet
        self.resnet_reformat_mat = [(siren_in, hidden_layer_dim)]
        for i in range(hidden_layer_width):
            self.resnet_reformat_mat.append((hidden_layer_dim, hidden_layer_dim))
        self.resnet_reformat_mat.append((hidden_layer_dim, siren_out))

        params = siren_in * hidden_layer_dim + hidden_layer_dim
        hidden_layer_width*hidden_layer_dim+siren_in+siren_out

        rng = tf.random.get_global_generator()
        rng.reset_from_seed(seed)

        self.Generator = ResNet(
            xlen, output_dim=params, lin_activation=tf.nn.leaky_relu, lin_output_activation=tf.nn.relu)
        self.i_omega = initial_omega
        self.hidden_omega = hidden_omega

    def __call__(self, siren_inputs, resnet_inputs):
        resnet_out = self.Generator(resnet_inputs)
        i = 0
        intermediate = siren_inputs
        for layer in self.resnet_reformat_mat[-1]:
            linear_layer = resnet_out[i:layer[0]*layer[1]] 
            i+= layer[0]+layer[1]
            bias_layer = resnet_out[i:layer[1]]
            i += layer[1]
            linear_layer = tf.reshape(linear_layer (layer[0], layer[1]))
            bias_layer = tf.reshape(bias_layer (1, layer[1]))
            intermediate = tf.math.sin(self.hidden_omega * (intermediate @ linear_layer + bias_layer))            
        ##Final layer is not sined
        linear_layer = resnet_out[i:layer[-1][0]*layer[-1][1]] 
        i+= layer[-1][0]+layer[-1][1]
        bias_layer = resnet_out[i:layer[-1][1]]
        i += layer[-1][1]
        linear_layer = tf.reshape(linear_layer (layer[-1][0], layer[-1][1]))
        bias_layer = tf.reshape(bias_layer (1, layer[-1][1]))
        intermediate = intermediate @ linear_layer + bias_layer            
        return intermediate

def get_image_tensor(image_path, xlen, ylen):
    img = Image.open(image_path)
    array = np.asarray(img)
    image = tf.image.resize(array, (xlen, ylen))
    # Image is now bound between 0 & 2
    image = (image / 128.0)
    # Image is now bound between -1 * 1
    image = image - 1
    return image



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from CIFAR_UTILS import render_img, unpickle

    from tqdm import trange
    from adam import Adam

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
    train_size = 10
    size = train_size
    images = np.reshape(raw_dict[b"data"], IMG_DIMS)
    images = np.transpose(images, (0, 2, 3, 1))

    model = Gen_Siren(input_dim=2, output_dim=3,
                  hidden_layer_dim=32, hidden_layer_width=3)

    # optimizer = Adam(size=len(model.trainable_variables), step_size=0.001)
    optimizer = tf.optimizers.AdamW(learning_rate=0.001)

    
    ##Train on one image at a time
    BATCH_SIZE = 1
    NUM_ITERS = 40
    

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
            loss = tf.math.reduce_mean((predicted - truth)**2)

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
    # Convert this image to a range from 0 - 2 & then 0 -1
    flattened = (flattened + 1) / 2
    render_img(flattened, "model_output.png")

    # loss = tf.math.reduce_mean((flattened - truth)**2)
    print("On Test Card F, achieved accuracy of %0.1f", loss.numpy())
