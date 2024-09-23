import tensorflow as tf
from linear import Linear

class Conv2d(tf.Module):
    def __init__(self,dim,in_channel=1,out_channel=1):
        rng = tf.random.get_global_generator()
        #Square kernel
        self.kernel = tf.Variable(
            rng.normal(shape=[dim,dim,in_channel,out_channel]),
            trainable=True
        )
        
    def __call__(self,input):
        tf.nn.conv2d(input,self.kernel,strides=1,padding="SAME")
        


class Classifier(tf.Module):
    pass


##Converts 1xn labels into nx10 labels with each index representing a 0
def restructure(labels):
    mat = np.zeros((labels.size,10))
    for x in range(0,labels.size):
        mat[x,labels[x]] = 1
    return mat




if __name__ == "__main__":
    from adam import Adam
    from tqdm import trange
    import numpy as np
    import os
    from load_mnist import load_data_arr

    # Getting training data from local MNIST path
    mnist_location = "MNIST"
    images_path = os.path.join(mnist_location, "train-images-idx3-ubyte.gz")
    labels_path = os.path.join(mnist_location, "train-labels-idx1-ubyte.gz")
    #Converting to float 32 as this is required input for a convolution
    images = load_data_arr(images_path).astype(np.float32)
    labels = load_data_arr(labels_path)

    images = load_data_arr(images_path).astype(np.float32)
    labels = load_data_arr(labels_path)

    BATCH_SIZE = 100
    NUM_ITERS = 1000
    VALIDATE = False
    VALIDATE_SPLIT = 0.8
    TEST = False
    tf.random.set_seed(42)
    rng = np.random.default_rng(seed=42)

    restruct_labels = restructure(labels)


    # Used to get random indexes for SGD
    max_ = np.arange(labels.size)

    model = Classifier()
    optimizer = Adam(size=len(model.trainable_variables))
    loss = tf.keras.losses.CategoricalCrossentropy()


    #Split training data down validate split
    size = int(VALIDATE_SPLIT * labels.size)
    bar = trange(NUM_ITERS)
    for i in bar:
        batch_indices = rng.integers(low=0, high=size, size=BATCH_SIZE).T
        with tf.GradientTape() as tape:
            batch_images = images[batch_indices, :]
            batch_labels = restruct_labels[batch_indices, :]
            predicted = model(batch_images)
            # Soft-max at the end
            # Cross Entropy Loss Function
            loss()

        grads = tape.gradient(loss,model.trainable_variables)
        optimizer.train(grads=grads,vars=model.trainable_variables)
        if i % 10 == 9:
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, learning_rate => {0.001}"
            )
            bar.refresh()


    if VALIDATE:
        validation_images = images[size:,:,:]
        validation_labels = labels[labels:]
        model_output = model(validation_images)
        right = np.sum(model_output == validation_labels)
        accuracy = right / validation_labels.size()
        print("On testing set, achieved accuracy of %d \%", accuracy)