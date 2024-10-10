import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def unpickle(file: str) -> dict:
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def render_img(image, path, label=""):
    path = os.path.join("testing_imgs", path)
    ##Testing to make sure images are generated properly
    plt.imshow(image)
    plt.ylabel(label)
    plt.savefig(path)


def augment(
    image_set: np.ndarray, render=False, batch="", labels=[0], label_strings=[""]
):
    ##Returns number of clones (for label expansion),
    ##as well as the modified images themselves
    mean = 0
    sigma = 3
    noise = np.random.normal(mean, sigma, image_set.shape)

    # exploiting uint8 for casting purposes & to add more noise
    noised = (noise + image_set).astype(np.uint8)
    ##Cutoffs
    flip_1 = np.flip(image_set, axis=1)
    flip_2 = np.flip(image_set, axis=2)
    # Because it's a uint8, this works

    invert = 0 - image_set
    if render:
        end = batch + ".png"
        render_img(image_set[0], "original_ " + end, label_strings[labels[0]])
        render_img(flip_1[0], "Flip1_" + end, label_strings[labels[0]])
        render_img(flip_2[0], "Flip2_" + end, label_strings[labels[0]])
        render_img(invert[0], "invert_" + end, label_strings[labels[0]])
        render_img(noised[0], "noised_" + end, label_strings[labels[0]])

    return np.concatenate((flip_1, flip_2, invert, noised)), 4


# Converts 1xn labels into nx10 labels with each index representing a 0
def restructure(labels):
    mat = np.zeros((labels.size, 10))
    for x in range(0, labels.size):
        mat[x, labels[x]] = 1
    return mat
