import tensorflow as tf
from linear import Linear
from sklearn.inspection import DecisionBoundaryDisplay


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

def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import trange
    import numpy as np
    # Constants for data (not using a config.yaml this time)
    SCALE = 5.4
    NUM_DATAPOINTS = 200
    THETA_DEV = 0.4
    THETA_INIT = 3 * np.pi / 5
    THETA_FINAL = 2 * np.pi * SCALE
    # The seed here is the same as the previous homework, but different generator?
    rng = np.random.default_rng(seed=47)
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
    #    rng.shuffle(dataset)
    # Model parameters
    batch_size = 100
    num_iters = 1000
    # step_size = 0.05
    # decay_rate = 0.999

    model = MLP(
        num_inputs=2,
        num_outputs=1,
        num_hidden_layers=5,
        hidden_layer_width=256,
        hidden_activation=tf.nn.leaky_relu,
        output_activation=tf.nn.sigmoid
    )        

    bar = trange(num_iters)
    ##using standard Adam implementation
    optimizer = tf._optimizers.Adam()
    epsilon = 1e-8
    for i in bar:
        batch_indices = rng.integers(
            low=0, high=dataset.shape[0], size=batch_size).T
        with tf.GradientTape() as tape:
            slice = np.take(dataset, batch_indices, axis=0)
            points = slice[:, :2]
            expected = slice[:, 2].reshape(batch_size,1)
            # print(x_batch)
            # print(y_batch)
            # print(expected)
            calculated = tf.cast(model(points), dtype=tf.float64)
            loss = tf.math.reduce_mean(
                -1 * (expected * tf.math.log(calculated+epsilon)
                      + (1 - expected) * tf.math.log(1-calculated + epsilon)))

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))
        # grad_update(step_size, model.trainable_variables,grads)
        # step_size *= decay_rate
        if i % 30 == (30 - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, learning_rate => {0.001}"
            )
            bar.refresh()

    fig, (ax1) = plt.subplots(1, 1)

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
    # ax1.plot(blue_x, blue_y, "bo")
    # ax1.plot(red_x, red_y, "ro")
    # ax1.set_xlabel("x")
    # ax1.set_ylabel("y")
    # ax1.set_title("Fig1")
    # h = ax1.set_ylabel("y", labelpad=10)
    # h.set_rotation(0)

    # fig.savefig("Submissions/plot.pdf")
    # plt.show()
