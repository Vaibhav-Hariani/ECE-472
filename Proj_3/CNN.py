import tensorflow as tf
import os
from load_mnist import load_data_arr


# My Little Perceptron
class CNN(tf.Module):
    pass


if __name__ == "__main__":
    mnist_location = "MNIST"
    images_path = os.path.join(mnist_location, "train-images-idx3-ubyte.gz")
    labels_path = os.path.join(mnist_location, "train-labels-idx1-ubyte.gz")
    training_images = load_data_arr(images_path)
    training_labels = load_data_arr(labels_path)
    print(training_labels)

