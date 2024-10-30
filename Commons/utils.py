import numpy as np


# Converts 1xN labels into NxSize labels with each expected value representing a 1
def restructure(labels, dim, num_ops=4):
    mat = np.zeros([dim, num_ops])
    len = np.arange(0, dim)
    mat[len, labels] = 1
    return mat
