import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file: str) -> dict:
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def render_img(image,path):
    ##Testing to make sure images are generated properly
    plt.imshow(image)  
    plt.savefig(path)

def augment(image_set: np.ndarray,render=False):
    mean = 0
    sigma = 3
    noise = np.random.normal(mean,sigma,image_set.shape)

    # exploiting uint8 for casting purposes & to add more noise
    noised = (noise+image_set).astype(np.uint8)
    ##Cutoffs
    flip_1 = np.flip(image_set,axis=1)
    flip_2 = np.flip(image_set, axis=2)

    #Because it's a uint8, this works
    invert = 0 - image_set
    if render:
        render_img(image_set[0], "original.png")
        render_img(flip_1[0], "Flip1.png")
        render_img(flip_2[0], "Flip2.png")
        render_img(invert[0], "invert.png")
        render_img(noised[0], "noised.png")

    return np.concatenate((flip_1,flip_2,invert,noised))
