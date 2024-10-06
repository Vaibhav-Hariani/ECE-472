import math

import tensorflow as tf
from conv2d import Conv2d
from MLP import MLP

class GroupNorm(tf.Module):
    pass

class ResidualBlock(tf.Module):
    pass

class Classifier(tf.Module):
    def __init__(
        self,
        input_dims,
        output_dim,
        conv_dims=3,
        num_convs=4,
        num_lin_layers=5,
        hidden_lin_width=125,
        lin_activation=tf.identity,
        lin_output_activation=tf.identity,
        dropout_rate=0.1,
    ):
        self.input_dims = input_dims
        self.convs = []
        for i in range(0, num_convs):
            self.convs.append(Conv2d(conv_dims, dropout_rate=dropout_rate))


        self.perceptron = MLP(
            num_inputs=input_dims**2,
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

        current = tf.reshape(current, (current.shape[0], -1))
        return self.perceptron(current, dropout)


if __name__ == "__main__":
    import os
    import numpy as np
    from CIFAR_UTILS import unpickle, augment
    from tqdm import trange

    import cv2

    # Getting training data from local CIFAR
    CIFAR_LOC = "CIFAR"
    CIFAR_FOLDER= "cifar-10-batches-py"
    IMG_DIMS=(-1,3,32,32)

    batches = ["data_batch_" + str(x) for x in range(1,2)]
    label_strings = unpickle(os.path.join(CIFAR_LOC,CIFAR_FOLDER, "batches.meta"))[b'label_names']
    images = []
    labels = []
    for batch in batches:
        path = os.path.join(CIFAR_LOC, CIFAR_FOLDER,batch)
        raw_dict = unpickle(path)
        batch_images = np.reshape(raw_dict[b'data'], IMG_DIMS)
        batch_images = np.transpose(batch_images,(0,2,3,1))        
        images.append(batch_images)
        ##Images are subject to gaussian noise, inversion, and flipping in two dimensions
        extra_images = augment(batch_images)
        images.append(extra_images)
        labels.append(raw_dict[b'labels']) 

    image_set = np.concatenate(images, axis=0)
    labels=np.concatenate(images,axis=0)
    
