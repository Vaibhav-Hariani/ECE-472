import math

import tensorflow as tf
from adam import Adam
from conv2d import Conv2d
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
            rng.normal(shape=[1, 1, 1, channels], stddev=stddev),
            trainable=True,
            name="GN/beta",
        )
        self.eps = eps
        self.groups = groups

    # Directly from the paper
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
        # Using this to prevent soft copying
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
        # Flattens for perceptron
        return self.output_perceptron(perceptron_in, dropout)


if __name__ == "__main__":
    import os

    import numpy as np
    from CIFAR_UTILS import augment, render_img, restructure, unpickle
    from sklearn.metrics import top_k_accuracy_score
    from tqdm import trange

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    # Getting training data from local CIFAR
    CIFAR_LOC = "CIFAR"
    CIFAR_FOLDER = "cifar-100-python"
    IMG_DIMS = (-1, 3, 32, 32)

    TRAIN_SET = "train"
    label_strings = unpickle(os.path.join(CIFAR_LOC, CIFAR_FOLDER, "meta"))[
        b"fine_label_names"
    ]
    path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, TRAIN_SET)
    raw_dict = unpickle(path)
    print("Opening " + TRAIN_SET)

    VALIDATE = True
    VALIDATE_SPLIT = 0.95
    image_list = []
    label_list = []

    train_size = int(VALIDATE_SPLIT * len(raw_dict[b"fine_labels"]))
    images = np.reshape(raw_dict[b"data"], IMG_DIMS)

    # This line is necessary for visualizing and rendering, as we expect channels at the back
    images = np.transpose(images, (0, 2, 3, 1))
    train_images = images[:train_size]
    validation_images = images[train_size:]
    train_labels = raw_dict[b"fine_labels"][:train_size]
    validation_labels = np.array(raw_dict[b"fine_labels"][train_size:])

    # Images are subject to gaussian noise, inversion, and flipping in two dimensions
    num_clones = 0
    extra_images, num_clones = augment(train_images)
    # Shuffling so that validation set is representative
    image_list.append(train_images)
    image_list.append(extra_images)
    label_list.append(train_labels * (num_clones + 1))

    train_images = np.concatenate(image_list, axis=0)
    train_labels = np.concatenate(label_list, axis=0).reshape(-1, 1)

    restruct_labels = restructure(train_labels, 100)

    TEST = True

    size = train_size * (num_clones + 1)

    # ##generating a set of labelled images.
    # k = 5
    # indexes = np.random.randint(0, train_labels.size, k)
    # image_slice = train_images[indexes]
    # label_slice = train_labels[indexes]
    # for i in range(k):
    #     render_img(image=images[i],path=str(i), label=label_strings[train_labels[i]])

    BATCH_SIZE = 128
    NUM_ITERS = 40
    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    print("Making Model")

    model = Classifier(
        input_dims=32,
        output_dim=100,
        pool_dims=2,
        conv_dims=3,
        conv_activation=tf.nn.leaky_relu,
        num_lin_layers=5,
        hidden_lin_width=125,
        lin_activation=tf.nn.leaky_relu,
        lin_output_activation=tf.nn.softmax,
        dropout_rate=0.0,
        group_sizes=[3, 5, 8, 16, 32, 32, 32, 1],
        channel_scales=[3, 5, 16, 32, 64, 128, 256, 1],
    )

    optimizer = tf.optimizers.AdamW(learning_rate=0.001)

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
            batch_images = tf.cast(train_images[batch_indices], dtype=tf.float32)
            # batch_images = tf.expand_dims(image_slice, axis=3)

            batch_labels = restruct_labels[batch_indices, :]
            predicted = model(batch_images, True)
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
        if i % 3 == 0:
            if i % 240 == 0:
                # Mini validation to see performance
                model_output = model(validation_images)
                top_5_accuracy = 100 * top_k_accuracy_score(
                    validation_labels, model_output, k=5
                )
            bar.set_description(
                f"epoch {epochs:0.4f}; Loss => {loss.numpy():0.4f}, Top-5 accuracy => {top_5_accuracy:0.2f}:"
            )
            bar.refresh()

    model_path = os.path.join("Proj_4", "CIFAR_100/1/")
    tf.saved_model.save(model, model_path)
    model_out = model(validation_images)
    print(
        "On validation set, achieved Top-1 accuracy of %0.1f%%"
        % (100 * top_k_accuracy_score(validation_labels, model_out, k=1))
    )
    print(
        "On validation set, achieved Top-5 accuracy of %0.1f%%"
        % (100 * top_k_accuracy_score(validation_labels, model_out, k=5))
    )
    # fig, ax1 = plt.subplots(1, 1)

    if TEST:
        dict_path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, "test")
        raw_dict = unpickle(dict_path)
        test_images = np.reshape(raw_dict[b"data"], IMG_DIMS)
        # This line is necessary for visualizing and rendering, as we expect channels at the back
        test_images = np.transpose(test_images, (0, 2, 3, 1)).astype(np.float32)[:500]
        # ##Images are subject to gaussian noise, inversion, and flipping in two dimensions
        # extra_images,num_clones = augment(batch_images)
        test_labels = np.array(raw_dict[b"fine_labels"])[:500]
        model_out = model(test_images)
        print(
            "On test set, achieved Top-1 accuracy of %0.1f%%"
            % (100 * top_k_accuracy_score(test_labels, model_out, k=1))
        )
        print(
            "On test set, achieved Top-5 accuracy of %0.1f%%"
            % (100 * top_k_accuracy_score(test_labels, model_out, k=5))
        )


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
        # Flattens for perceptron
        return self.output_perceptron(perceptron_in, dropout)


if __name__ == "__main__":
    import os

    import numpy as np
    from CIFAR_UTILS import augment, render_img, restructure, unpickle
    from sklearn.metrics import top_k_accuracy_score
    from tqdm import trange

    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    # Getting training data from local CIFAR
    CIFAR_LOC = "CIFAR"
    CIFAR_FOLDER = "cifar-100-python"
    IMG_DIMS = (-1, 3, 32, 32)

    TRAIN_SET = "train"
    label_strings = unpickle(os.path.join(CIFAR_LOC, CIFAR_FOLDER, "meta"))[
        b"fine_label_names"
    ]
    path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, TRAIN_SET)
    raw_dict = unpickle(path)
    print("Opening " + TRAIN_SET)

    VALIDATE = True
    VALIDATE_SPLIT = 0.95
    image_list = []
    label_list = []

    train_size = int(VALIDATE_SPLIT * len(raw_dict[b"fine_labels"]))
    images = np.reshape(raw_dict[b"data"], IMG_DIMS)

    # This line is necessary for visualizing and rendering, as we expect channels at the back
    images = np.transpose(images, (0, 2, 3, 1))
    train_images = images[:train_size]
    validation_images = images[train_size:]
    train_labels = raw_dict[b"fine_labels"][:train_size]
    validation_labels = np.array(raw_dict[b"fine_labels"][train_size:])

    # Images are subject to gaussian noise, inversion, and flipping in two dimensions
    num_clones = 0
    extra_images, num_clones = augment(train_images)
    # Shuffling so that validation set is representative
    image_list.append(train_images)
    image_list.append(extra_images)
    label_list.append(train_labels * (num_clones + 1))

    train_images = np.concatenate(image_list, axis=0)
    train_labels = np.concatenate(label_list, axis=0).reshape(-1, 1)

    restruct_labels = restructure(train_labels, 100)

    TEST = True

    size = train_size * (num_clones + 1)

    # ##generating a set of labelled images.
    # k = 5
    # indexes = np.random.randint(0, train_labels.size, k)
    # image_slice = train_images[indexes]
    # label_slice = train_labels[indexes]
    # for i in range(k):
    #     render_img(image=images[i],path=str(i), label=label_strings[train_labels[i]])

    BATCH_SIZE = 128
    NUM_ITERS = 40
    tf_rng = tf.random.get_global_generator()
    tf_rng.reset_from_seed(42)
    np_rng = np.random.default_rng(seed=42)

    print("Making Model")

    model = Classifier(
        input_dims=32,
        output_dim=100,
        pool_dims=2,
        conv_dims=3,
        conv_activation=tf.nn.leaky_relu,
        num_lin_layers=5,
        hidden_lin_width=125,
        lin_activation=tf.nn.leaky_relu,
        lin_output_activation=tf.nn.softmax,
        dropout_rate=0.0,
        group_sizes=[3, 5, 8, 16, 32, 32, 32, 1],
        channel_scales=[3, 5, 16, 32, 64, 128, 256, 1],
    )

    optimizer = tf.optimizers.AdamW(learning_rate=0.001)

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
            batch_images = tf.cast(train_images[batch_indices], dtype=tf.float32)
            # batch_images = tf.expand_dims(image_slice, axis=3)

            batch_labels = restruct_labels[batch_indices, :]
            predicted = model(batch_images, True)
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
        if i % 3 == 0:
            if i % 240 == 0:
                # Mini validation to see performance
                model_output = model(validation_images)
                top_5_accuracy = 100 * top_k_accuracy_score(
                    validation_labels, model_output, k=5
                )
            bar.set_description(
                f"epoch {epochs:0.4f}; Loss => {loss.numpy():0.4f}, Top-5 accuracy => {top_5_accuracy:0.2f}:"
            )
            bar.refresh()

    model_path = os.path.join("Proj_4", "CIFAR_100/1/")
    tf.saved_model.save(model, model_path)
    model_out = model(validation_images)
    print(
        "On validation set, achieved Top-1 accuracy of %0.1f%%"
        % (100 * top_k_accuracy_score(validation_labels, model_out, k=1))
    )
    print(
        "On validation set, achieved Top-5 accuracy of %0.1f%%"
        % (100 * top_k_accuracy_score(validation_labels, model_out, k=5))
    )
    # fig, ax1 = plt.subplots(1, 1)

    if TEST:
        dict_path = os.path.join(CIFAR_LOC, CIFAR_FOLDER, "test")
        raw_dict = unpickle(dict_path)
        test_images = np.reshape(raw_dict[b"data"], IMG_DIMS)
        # This line is necessary for visualizing and rendering, as we expect channels at the back
        test_images = np.transpose(test_images, (0, 2, 3, 1)).astype(np.float32)[:500]
        # ##Images are subject to gaussian noise, inversion, and flipping in two dimensions
        # extra_images,num_clones = augment(batch_images)
        test_labels = np.array(raw_dict[b"fine_labels"])[:500]
        model_out = model(test_images)
        print(
            "On test set, achieved Top-1 accuracy of %0.1f%%"
            % (100 * top_k_accuracy_score(test_labels, model_out, k=1))
        )
        print(
            "On test set, achieved Top-5 accuracy of %0.1f%%"
            % (100 * top_k_accuracy_score(test_labels, model_out, k=5))
        )
