import math

import tensorflow as tf
from adam import Adam
from conv2d import Conv2d
from MLP import MLP


class GroupNorm(tf.Module):
    def __init__(self, groups, channels, eps=1e-5, seed=42):
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(seed)
        stddev = 2 / channels
        self.gamma = tf.Variable(
            rng.normal(shape=[1, 1, 1, channels], stddev=stddev),
            trainable=True,
            name="GN/gamma",
        )
        self.beta = tf.Variable(
            rng.normal(shape=[1, 1, 1, channels], stddev=stddev),
            trainable=True,
            name="GN/beta",
        )
        self.eps = eps
        self.groups = groups

    ##Directly from the paper
    def __call__(self, x):
        N, H, W, C = x.shape
        # C = channels
        x = tf.reshape(x, [N, self.groups, C // self.groups, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta


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
        ##Using this to prevent soft copying
        input = self.input_conv(input)
        intermediate = input
        for i in range(self.num_layers):
            intermediate = self.GNs[i](intermediate)
            intermediate = self.activation(intermediate)
            intermediate = self.convs[i](intermediate, dropout)

        return input + intermediate


class Classifier(tf.Module):
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
        ##Flattens for perceptron
        return self.output_perceptron(perceptron_in, dropout)


if __name__ == "__main__":
    import os

    import numpy as np
    from CIFAR_UTILS import augment, restructure, unpickle
    from tqdm import trange

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    # Getting training data from local CIFAR
    CIFAR_LOC = "CIFAR"
    CIFAR_FOLDER = "cifar-10-batches-py"
    IMG_DIMS = (-1, 3, 32, 32)

    batches = ["data_batch_" + str(x + 1) for x in range(4)]
    label_strings = unpickle(
        os.path.join(CIFAR_LOC, CIFAR_FOLDER, "batches.meta")
    )[b"label_names"]
    images = []
    labels = []
    for batch in batches:
        path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, batch)
        raw_dict = unpickle(path)
        print("Opening batch " + batch)
        batch_images = np.reshape(raw_dict[b"data"], IMG_DIMS)
        # This line is necessary for visualizing and rendering, as we expect channels at the back
        batch_images = np.transpose(batch_images, (0, 2, 3, 1))
        images.append(batch_images)
        ##Images are subject to gaussian noise, inversion, and flipping in two dimensions
        num_clones = 0
        extra_images, num_clones = augment(batch_images)
        ##Shuffling so that validation set is representative
        images.append(extra_images)
        labels.append(raw_dict[b"labels"] * (num_clones + 1))

    VALIDATE = True
    VALIDATE_SPLIT = 0.95
    TEST = True

    ##Creating validation set out of a slice of the last batch: This is 500 images.
    dict_path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, "data_batch_5")
    raw_dict = unpickle(path)
    test_images = np.reshape(raw_dict[b"data"], IMG_DIMS)
    split_len = int(len(raw_dict[b"labels"]) * VALIDATE_SPLIT)
    batch_5_labels = raw_dict[b"labels"]
    batch_5_imgs = np.transpose(test_images, (0, 2, 3, 1)).astype(np.float32)

    images.append(batch_5_imgs[:split_len])
    extra_images, num_clones = augment(batch_5_imgs[:split_len])
    images.append(extra_images)
    labels.append(batch_5_labels[:split_len] * (num_clones + 1))

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    restruct_labels = restructure(labels)

    validation_images = batch_5_imgs[split_len:]
    validation_labels = np.array(batch_5_labels)[split_len:]

    # ##generating a set of labelled images.
    # k = 5
    # indexes = np.random.randint(0, labels.size, k)
    # image_slice = images[indexes]
    # label_slice = labels[indexes]
    # for i in range(k):
    #     render_img(image=images[i],path=str(i), label=label_strings[labels[i]])

    BATCH_SIZE = 128
    size = int(labels.size * VALIDATE_SPLIT)

    # NUM_ITERS = 4000
    epochs = 0
    total_epochs = 20
    NUM_ITERS = int(total_epochs * size / BATCH_SIZE)

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    print("Making Model")

    model = Classifier(
        input_dims=32,
        output_dim=10,
        pool_dims=2,
        conv_dims=3,
        conv_activation=tf.nn.leaky_relu,
        num_lin_layers=5,
        hidden_lin_width=125,
        lin_activation=tf.nn.leaky_relu,
        lin_output_activation=tf.nn.softmax,
        dropout_rate=0.05,
        group_sizes=[3, 5, 8, 16, 32, 3, 1],
        channel_scales=[3, 5, 16, 32, 64, 3, 1],
    )

    # optimizer = Adam(size=len(model.trainable_variables), step_size=0.001)
    optimizer = tf.optimizers.AdamW(learning_rate=0.001)

    ##Converting batch_size to epochs
    epochs = 0
    total_epochs = BATCH_SIZE * NUM_ITERS / size

    print("Running for %0.4f epochs" % total_epochs)
    print("Running for %0.4f iterations" % NUM_ITERS)

    n_min = 0.1
    n_max = 2

    bar = trange(NUM_ITERS)
    for i in bar:
        batch_indices = np_rng.integers(low=0, high=size, size=BATCH_SIZE).T
        with tf.GradientTape() as tape:
            batch_images = tf.cast(images[batch_indices], dtype=tf.float32)
            # batch_images = tf.expand_dims(image_slice, axis=3)

            batch_labels = restruct_labels[batch_indices, :]
            predicted = model(batch_images, True)
            # Cross Entropy Loss Function
            loss = tf.keras.losses.categorical_crossentropy(
                batch_labels, predicted
            )
            loss = tf.math.reduce_mean(loss)

        epochs += BATCH_SIZE / size
        # ##Cosine annealing
        # n_t = n_min + (n_max - n_min) * (
        #     1 + tf.math.cos(epochs * math.pi / total_epochs)
        # )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # optimizer.train(
        #     grads=grads, vars=model.trainable_variables, adamW=True, decay_scale=0.1
        # )
        if i % 3 == 0:
            if i % 240 == 0:
                ##Mini validation to see performance
                model_output = np.argmax(model(validation_images), axis=1)
                accuracy = np.sum(model_output == validation_labels) / (
                    validation_labels.size
                )
            bar.set_description(
                f"epoch {epochs:0.4f}; Loss => {loss.numpy():0.4f}, accuracy => {accuracy:0.3f}:"
            )
            bar.refresh()

    # model_output = np.argmax(model(validation_images), axis=1)
    # accuracy = np.sum(model_output == validation_labels) / validation_labels.size
    # print("On validation set, achieved accuracy of %.1f%%" % (100 * accuracy))
    model_output = np.argmax(model(validation_images), axis=1)
    accuracy = (
        np.sum(model_output == validation_labels) / validation_labels.size
    )
    print("On validation set, achieved accuracy of %.1f%%" % (100 * accuracy))

    # fig, ax1 = plt.subplots(1, 1)
    if TEST:
        dict_path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, "test_batch")
        raw_dict = unpickle(path)
        test_images = np.reshape(raw_dict[b"data"], IMG_DIMS)
        # This line is necessary for visualizing and rendering, as we expect channels at the back
        test_images = np.transpose(test_images, (0, 2, 3, 1)).astype(np.float32)
        # ##Images are subject to gaussian noise, inversion, and flipping in two dimensions
        # extra_images,num_clones = augment(batch_images)
        test_labels = np.array(raw_dict[b"labels"])
        model_output = np.argmax(model(test_images), axis=1)
        accuracy = np.sum(model_output == test_labels) / test_labels.size
        print("On test set, achieved accuracy of %0.1f%%" % (100 * accuracy))
