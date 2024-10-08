import math

import tensorflow as tf
from conv2d import Conv2d
from adam import Adam
from MLP import MLP


class GroupNorm(tf.Module):
    def __init__(self, groups, channels, eps=1e-5, seed=42):
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(seed)
        stddev = 1 / channels
        self.gamma = tf.Variable(
            rng.normal(shape=[1, 1, 1, channels], stddev=stddev),
            trainable=True,
            name="GN/gamma",
        )
        self.beta = tf.Variable(
            rng.normal(shape=[1, 1, 1,channels], stddev=stddev),
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
        layers_between=2,
        in_channel=3,
        out_channel=3
        ):
        self.res_kernel = [dim,dim,in_channel,out_channel]
        self.GNs = []
        self.convs = []
        for layer in range(layers_between):
            self.convs.append(
                Conv2d(dim=dim, in_channel=in_channel, out_channel=out_channel, dropout_rate=dropout_rate)
            )
            self.GNs.append(GroupNorm(groups=groups, channels=in_channel))

        self.activation = activation

    def __call__(self, input, dropout=False):
        ##Using this to prevent soft copying
        intermediate = tf.identity(input)


        for layer, gn in (self.convs, self.GNs):
            intermediate = gn(intermediate)
            intermediate = self.activation(intermediate)
            intermediate = layer(intermediate, dropout)

        ##Upsampling, to allow for differently shaped convolutional layers            
        input = tf.nn.conv2d(input,self.res_kernel,strides=1,padding="SAME")

        return input + intermediate


class Classifier(tf.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        pool_dims=1,
        conv_dims=5,
        num_lin_layers=5,
        batch_norm_groups=3,
        hidden_lin_width=125,
        conv_activation=tf.identity,
        lin_activation=tf.identity,
        lin_output_activation=tf.identity,
        dropout_rate=0.1,
        channel_scales = [3,3,3,3,3]
    ):
    #Channel scales represents how the channels shift throughout
    #This encodes the input channel, all the way up to the output channel
        self.input_dims = input_dims
        self.res_layers = []

        for res_layer in range(0, channel_scales-1):
            self.res_layers.append(
                ResidualBlock(
                    dim=conv_dims,
                    in_channel=channel_scales[res_layer],
                    out_channel=channel_scales[res_layer+1],
                    groups=batch_norm_groups,
                    activation=conv_activation,
                    dropout_rate=dropout_rate,
                )
            )
        self.pool_kernel = [1, pool_dims, pool_dims, 1]
        self.strides = [1, pool_dims, pool_dims, 1]
        self.padding = "SAME"

        perceptron_dims = (input_dims * input_dims * channel_scales[-1]) // (pool_dims*pool_dims)
        self.output_perceptron = MLP(
            num_inputs= perceptron_dims,
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

        pooled_out = tf.nn.max_pool2d(
            current, ksize=self.pool_kernel, strides=self.strides, padding=self.padding
        )
        perceptron_in = tf.reshape(pooled_out, (pooled_out.shape[0], -1))

        ##Flattens for perceptron
        return self.output_perceptron(perceptron_in, dropout)


if __name__ == "__main__":
    import os
    import numpy as np
    from CIFAR_UTILS import unpickle, augment, restructure
    from tqdm import trange

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)


    # Getting training data from local CIFAR
    CIFAR_LOC = "CIFAR"
    CIFAR_FOLDER = "cifar-10-batches-py"
    IMG_DIMS = (-1, 3, 32, 32)

    batches = ["data_batch_" + str(x + 1) for x in range(4)]
    label_strings = unpickle(os.path.join(CIFAR_LOC, CIFAR_FOLDER, "batches.meta"))[
        b"label_names"
    ]
    images = []
    labels = []
    for batch in batches:
        path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, batch)
        raw_dict = unpickle(path)
        print("Opening batch " + batch)
        batch_images = np.reshape(raw_dict[b"data"], IMG_DIMS)
        #This line is necessary for visualizing and rendering, as we expect channels at the back
        batch_images = np.transpose(batch_images, (0, 2, 3,  1))
        images.append(batch_images)
        ##Images are subject to gaussian noise, inversion, and flipping in two dimensions
        num_clones=0
        extra_images,num_clones = augment(batch_images)
        ##Shuffling so that validation set is representative
        images.append(extra_images)
        labels.append(raw_dict[b"labels"]*(num_clones+1))

    images = np.concatenate(images, axis=0)
    ##shuffling with a seed to 
    labels = np.concatenate(labels, axis=0)

    # randomize = np.arange(len(labels))
    # np_rng.shuffle(randomize)

    # ##Shuffling them to create a reflective validation set.
    # images = images[randomize]
    # labels = labels[randomize]



    BATCH_SIZE = 250
    NUM_ITERS = 10000
    VALIDATE = True
    VALIDATE_SPLIT=1
    # VALIDATE_SPLIT = 0.95
    TEST = True
    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)
    size = int(labels.size * VALIDATE_SPLIT)

    dict_path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, "data_batch_5")
    raw_dict = unpickle(path)
    test_images = np.reshape(raw_dict[b"data"], IMG_DIMS)
    validation_images= np.transpose(test_images, (0, 2, 3, 1)).astype(np.float32)
    validation_labels = np.array(raw_dict[b"labels"])

    restruct_labels = restructure(labels)

    print("Making Model")
    
    model = Classifier(
        input_dims=32,
        output_dim=10,
        pool_dims=4,
        conv_dims=5,
        conv_activation=tf.nn.relu,
        num_lin_layers=5,
        hidden_lin_width=125,
        lin_activation=tf.nn.leaky_relu,
        lin_output_activation=tf.nn.softmax,
        dropout_rate=0.2,
        channel_scales=[3,15,30,15]
    )

    optimizer = Adam(size=len(model.trainable_variables), step_size=0.001)
    # Split training data down validate split
    bar = trange(NUM_ITERS)
    accuracy = 0

    ##Converting batch_size to epochs
    epochs = 0
    total_epochs = BATCH_SIZE * NUM_ITERS / size

    n_min = 0.1
    n_max = 2

    for i in bar:
        batch_indices = np_rng.integers(low=0, high=size, size=BATCH_SIZE).T
        with tf.GradientTape() as tape:
            batch_images = tf.cast(images[batch_indices], dtype=tf.float32)
            # batch_images = tf.expand_dims(image_slice, axis=3)

            batch_labels = restruct_labels[batch_indices, :]
            predicted = model(batch_images, True)
            # Cross Entropy Loss Function
            loss = tf.keras.losses.categorical_crossentropy(batch_labels, predicted)
            loss = tf.math.reduce_mean(loss)

        epochs += BATCH_SIZE / size
        ##Cosine annealing
        n_t = n_min + (n_max - n_min) * (
            1 + tf.math.cos(epochs * math.pi / total_epochs)
        )
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.train(
            grads=grads, vars=model.trainable_variables, adamW=True, decay_scale=n_t
        )
        if i % 10 == 0:
            if(i % 500 == 0):
                ##Mini validation to see performance
                model_output = np.argmax(model(validation_images[:validation_labels.size//5]), axis=1)
                accuracy = (
                    np.sum(model_output == validation_labels[:validation_labels.size//5]) / (validation_labels.size//5)
                )
                print(f"Step {i}; Loss => {loss.numpy():0.4f}, sample accuracy => {accuracy:0.3f}:")
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, sample accuracy => {accuracy:0.3f}:"
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
        #This line is necessary for visualizing and rendering, as we expect channels at the back
        test_images= np.transpose(test_images, (0, 2, 3, 1)).astype(np.float32)
        # ##Images are subject to gaussian noise, inversion, and flipping in two dimensions
        # extra_images,num_clones = augment(batch_images)
        test_labels = np.array(raw_dict[b"labels"])
        model_output = np.argmax(model(test_images), axis=1)
        accuracy = (np.sum(model_output == test_labels) / test_labels.size)
        print("On test set, achieved accuracy of %0.1f%%" % (100 * accuracy))

