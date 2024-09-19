import tensorflow as tf
from linear import Linear
from sklearn.inspection import DecisionBoundaryDisplay
from adam import Adam

#My Little Pony
class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation=tf.identity,
        output_activation=tf.identity,
    ):

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        # In this project, 1 output (Binary Classifier)
        # num inputs is 2: x coordinate & y coordinate
        self.linear_steps = []
        if num_hidden_layers == 0:
            lin_obj = Linear(num_inputs, num_outputs)
            self.linear_steps.append(lin_obj)

        else:
            obj1 = Linear(num_inputs, hidden_layer_width)
            self.linear_steps.append(obj1)
            for x in range(0, num_hidden_layers-1):
                lin_obj = Linear(hidden_layer_width, hidden_layer_width)
                self.linear_steps.append(lin_obj)
            final_obj = Linear(hidden_layer_width, num_outputs)
            self.linear_steps.append(final_obj)

    def __call__(self, layer_in):
        current = layer_in
        for i in self.linear_steps[:-1]:
            current = i(current)
            current = self.hidden_activation(current)
        current = self.linear_steps[-1](current)
        return self.output_activation(current)


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
    # Constants for data (not using a config.yaml this time)
    SCALE = 5.4
    NUM_DATAPOINTS = 200
    THETA_DEV = 0.3
    THETA_INIT = 3 * np.pi / 5
    THETA_FINAL = 2 * np.pi * SCALE
    # The seed here is the same as the previous homework, but different generator?
    rng = np.random.default_rng(seed=42)
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

    expected_value = np.full(blue_x.shape, 0)
    combined_blue = np.vstack((blue_x, blue_y, expected_value)).T

    expected_value.fill(1)
    combined_red = np.vstack((red_x, red_y, expected_value)).T

    # Combines the data
    dataset = np.concatenate((combined_red, combined_blue))
    BATCH_SIZE = 100
    NUM_ITERS = 1000

    model = MLP(
        num_inputs=2,
        num_outputs=1,
        num_hidden_layers=5,
        hidden_layer_width=256,
        hidden_activation=tf.nn.leaky_relu,
        output_activation=tf.nn.sigmoid
    )        

    bar = trange(NUM_ITERS)
    #Custom adam implementation
    optimizer = Adam(size=len(model.trainable_variables))
    epsilon = 1e-8
    for i in bar:
        batch_indices = rng.integers(
            low=0, high=dataset.shape[0], size=BATCH_SIZE).T
        with tf.GradientTape() as tape:
            slice = np.take(dataset, batch_indices, axis=0)
            points = slice[:, :2]
            expected = slice[:, 2].reshape(BATCH_SIZE,1)
            calculated = tf.cast(model(points), dtype=tf.float64)
            loss = tf.math.reduce_mean(
                -1 * (expected * tf.math.log(calculated+epsilon)
                      + (1 - expected) * tf.math.log(1-calculated + epsilon)))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.train(grads=grads,vars=model.trainable_variables)
        if i % 3 == 2:
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, learning_rate => {0.001}"
            )
            bar.refresh()

    X, Y = np.meshgrid(
        np.linspace(dataset[:, 0].min(), dataset[:, 0].max()),
        np.linspace(dataset[:, 1].min(), dataset[:, 1].max()))
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    pred = np.reshape(model(positions), X.shape)
    display = DecisionBoundaryDisplay(xx0=X, xx1=Y, response=pred)
    display.plot()
    display.ax_.scatter(red_x, red_y, c="r")
    display.ax_.scatter(blue_x, blue_y, c="b")
    display.figure_.savefig('Submissions/output_hw2.png')