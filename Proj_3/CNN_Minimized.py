import tensorflow as tf
from adam import Adam
from mlp import MLP


class Conv2d(tf.Module):
    def __init__(self, dim, in_channel=1, out_channel=1, dropout_rate=0, strides=1):
        # if out_channel == None:
        #     out_channel = dim
        rng = tf.random.get_global_generator()
        # Square kernel
        self.kernel = tf.Variable(
            rng.normal(shape=[dim, dim, in_channel, out_channel]), trainable=True
        )
        self.dropout_rate = dropout_rate
        self.strides = strides

    def __call__(self, input, dropout=False):
        conv = tf.nn.conv2d(input, self.kernel, strides=self.strides, padding="SAME")
        if dropout:
            return tf.nn.dropout(conv, self.dropout_rate)
        return conv


class Classifier(tf.Module):

    def __init__(
        self,
        input_dims,
        output_dim,
        conv_dims=4,
        num_convs=1,
        num_lin_layers=1,
        hidden_lin_width=10,
        lin_activation=tf.identity,
        lin_output_activation=tf.identity,
        dropout_rate=0.1,
        pool_size=0,
    ):
        self.input_dims = input_dims
        self.convs = []
        for i in range(0, num_convs):
            self.convs.append(Conv2d(conv_dims, dropout_rate=dropout_rate))

        self.pool_size = pool_size
        ##Hard Coded for this section: didn't want to have to deal with parametrizing the number of inputs
        self.perceptron = MLP(
            num_inputs=(input_dims // pool_size) ** 2,
            num_outputs=output_dim,
            num_hidden_layers=num_lin_layers,
            hidden_layer_width=hidden_lin_width,
            hidden_activation=lin_activation,
            output_activation=lin_output_activation,
            dropout_rate=dropout_rate,
        )

    def __call__(self, input, dropout=False):
        current = input
        for conv in self.convs:
            current = conv(current, dropout)

        if self.pool_size != 0:
            current = tf.nn.avg_pool2d(
                current, self.pool_size, strides=self.pool_size, padding="SAME"
            )

        current = tf.reshape(current, (current.shape[0], -1))
        return self.perceptron(current, dropout)


# Converts 1xn labels into nx10 labels with each index representing a 0
def restructure(labels):
    mat = np.zeros((labels.size, 10))
    for x in range(0, labels.size):
        mat[x, labels[x]] = 1
    return mat


if __name__ == "__main__":
    import os

    import numpy as np
    from adam import Adam
    from load_mnist import load_data_arr
    from tqdm import trange

    # Getting training data from local MNIST path
    mnist_location = "MNIST"
    images_path = os.path.join(mnist_location, "train-images-idx3-ubyte.gz")
    labels_path = os.path.join(mnist_location, "train-labels-idx1-ubyte.gz")
    # Converting to float 32 as this is required input for a convolution
    images = load_data_arr(images_path).astype(np.float32)
    labels = load_data_arr(labels_path)

    images = load_data_arr(images_path).astype(np.float32)
    labels = load_data_arr(labels_path)
    # images = images.T

    ##Tested for 96.3
    # BATCH_SIZE = 100
    # NUM_ITERS = 2500
    BATCH_SIZE = 100
    NUM_ITERS = 2500

    VALIDATE = True
    VALIDATE_SPLIT = 0.8
    TEST = True
    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)
    size = int(VALIDATE_SPLIT * labels.size)

    validation_images = tf.expand_dims(images[size:, :, :], axis=3)
    validation_labels = labels[size:]

    restruct_labels = restructure(labels)

    # Used to get random indexes for SGD/Adam
    max_ = np.arange(labels.size)

    model = Classifier(
        input_dims=28,
        output_dim=10,
        conv_dims=4,
        num_convs=3,
        num_lin_layers=5,
        hidden_lin_width=125,
        lin_activation=tf.nn.leaky_relu,
        lin_output_activation=tf.nn.softmax,
        dropout_rate=0.1,
        pool_size=2,
    )
    optimizer = Adam(size=len(model.trainable_variables), step_size=0.002)
    # Split training data down validate split
    bar = trange(NUM_ITERS)
    accuracy = 0

    for i in bar:
        batch_indices = np_rng.integers(low=0, high=size, size=BATCH_SIZE).T
        with tf.GradientTape() as tape:
            image_slice = images[batch_indices, :, :]
            batch_images = tf.expand_dims(image_slice, axis=3)

            batch_labels = restruct_labels[batch_indices, :]
            predicted = model(batch_images, True)
            # Soft-max at the end
            # Cross Entropy Loss Function
            loss = tf.keras.losses.categorical_crossentropy(batch_labels, predicted)
            loss = tf.math.reduce_mean(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.train(grads=grads, vars=model.trainable_variables, adamW=True)
        if i % 10 == 9:
            if i % 100 == 99:
                model_output = np.argmax(model(validation_images), axis=1)
                accuracy = (
                    np.sum(model_output == validation_labels) / validation_labels.size
                )
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, accuracy => {accuracy:0.3f}:"
            )
            bar.refresh()

    model_output = np.argmax(model(validation_images), axis=1)
    accuracy = np.sum(model_output == validation_labels) / validation_labels.size
    print("On validation set, achieved accuracy of %.1f%%" % (100 * accuracy))

    if TEST:
        images_path = os.path.join(mnist_location, "t10k-images-idx3-ubyte.gz")
        labels_path = os.path.join(mnist_location, "t10k-labels-idx1-ubyte.gz")

        images = load_data_arr(images_path).astype(np.float32)
        labels = load_data_arr(labels_path)

        images = tf.expand_dims(images, axis=3)
        model_output = np.argmax(model(images), axis=1)
        accuracy = np.sum(model_output == labels) / labels.size
        print("On test set, achieved accuracy of %0.1f %%" % (100 * accuracy))
