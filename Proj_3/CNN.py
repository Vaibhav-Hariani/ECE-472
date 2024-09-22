import tensorflow as tf
import os
from load_mnist import load_data_arr


# My Little Perceptron
class CNN(tf.Module):
    pass


if __name__ == "__main__":
    mnist_location = "MNIST"
    training_images = os.path.join(mnist_location, "train-images-idx3-ubyte.gz")
    training_labels = os.path.join(mnist_location, "train-labels-idx1-ubyte.gz")
    load_mnist()
