import tensorflow as tf
import linear

import tensorflow as tf



    

class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,):

        # In this project, 1 output (Binary Classifier)
        # num inputs is 2: x coordinate & y coordinate
        linear_steps = []
        if(num_hidden_layers == 0):
            lin_obj = Linear(num_inputs,num_outputs)
            linear_steps.append(lin_obj)
        else:
            obj1 = Linear(num_inputs,hidden_layer_width)
            linear_steps.append(obj1)
            for x in range(0,num_hidden_layers):
                lin_obj = Linear(hidden_layer_width,hidden_layer_width)
                linear_steps.append(obj1)
            final_obj = Linear(hidden_layer_width,num_outputs)


    def __call__(self, x):
        # Perform layered matrix multiplication, add bias at each step, and then return final outputs
        z = x @ self.w
        if self.bias:
            z += self.b

        return z


def random_spiral_gen(datapoints, dev, initial, final):
    # Realized your spirals are evenly spaced throughout
    # Found an equation to replicate this approximately
    linspace = np.linspace(initial, final, datapoints) / datapoints
    de_lin = np.sqrt(linspace) * final
    return (rng.normal(loc=de_lin, scale=dev), de_lin)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import trange
    import numpy as np

    NUM_LOOPS = 5.35
    NUM_DATAPOINTS = 200
    THETA_DEV = 0.3
    THETA_INIT = 3 * np.pi / 5
    THETA_FINAL = 2 * np.pi * NUM_LOOPS
    # The seed here is the same as the previous homework
    rng = np.random.default_rng(seed=0x43966E87BD57227011B5B03B58785EC1)
    blue_thetas, blue_points = random_spiral_gen(
        NUM_DATAPOINTS, THETA_DEV, THETA_INIT, THETA_FINAL
    )
    blue_x = -1 * np.cos(blue_thetas) * blue_points
    blue_y = 1 * np.sin(blue_thetas) * blue_points
    red_thetas, red_points = random_spiral_gen(
        NUM_DATAPOINTS, THETA_DEV, THETA_INIT, THETA_FINAL
    )

    red_x = np.cos(red_thetas) * red_points
    red_y = -1 * np.sin(red_thetas) * red_points

    fig, (ax1) = plt.subplots(1, 1)
    ax1.plot(blue_x, blue_y, "bo")
    ax1.plot(red_x, red_y, "ro")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Fig1")
    h = ax1.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    fig.savefig("plot.pdf")
