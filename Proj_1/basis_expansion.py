import tensorflow as tf


class Linear(tf.Module):
    def __init__(self, num_inputs, num_outputs, bias=True):
        rng = tf.random.get_global_generator()

        stddev = tf.math.sqrt(2 / (num_inputs + num_outputs))

        self.w = tf.Variable(
            rng.normal(shape=[num_inputs, num_outputs], stddev=stddev),
            trainable=True,
            name="Linear/w",
        )

        # Does not make sense looking at the data, but leaving it in
        self.bias = bias
        if self.bias:
            self.b = tf.Variable(
                tf.zeros(shape=[1, num_outputs]),
                trainable=True,
                name="Linear/b",
            )

    def __call__(self, x):
        z = x @ self.w

        if self.bias:
            z += self.b

        return z


class BasisExpansion(tf.Module):

    def __init__(self, M):
        self.linear = Linear(M, 1, bias=True)

        ## Trainable mu & sigmas
        self.mu = tf.Variable(
            rng.normal(shape=[1, M]),
            trainable=True,
            name="BasisExpansion/mu",
        )

        self.sig = tf.Variable(
            rng.normal(shape=[1, M]),
            trainable=True,
            name="BasisExpansion/sigma",
        )

    def __call__(self, x):
        ## This computes the PHI function
        phi = -1 * tf.math.square((x - self.mu) / self.sig)
        phi = tf.math.exp(phi)
        # Multiplies by weights for w
        z = self.linear(phi)
        return z

    ##Returns basis at a given index, for a set of values x
    def sliver_call(self, index, x):
        phi = -1 * tf.math.square((x - self.mu[0, index]) / self.sig[0, index])
        return tf.math.exp(phi)


def grad_update(step_size, variables, grads):
    for var, grad in zip(variables, grads):
        var.assign_sub(step_size * grad)


if __name__ == "__main__":
    import argparse

    from pathlib import Path

    import matplotlib.pyplot as plt
    import yaml

    from tqdm import trange

    parser = argparse.ArgumentParser(
        prog="Linear",
        description="Fits a linear model to some data, given a config",
    )

    parser.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(0x43966E87BD57227011B5B03B58785EC1)

    num_samples = config["data"]["num_samples"]
    M = 10

    #
    x = rng.uniform(shape=(num_samples, 1))

    y = rng.normal(
        shape=(num_samples, 1),
        mean=tf.sin(2 * 3.14159 * x),
        stddev=config["data"]["noise_stddev"],
    )

    basis = BasisExpansion(M)

    num_iters = config["learning"]["num_iters"]
    step_size = config["learning"]["step_size"]
    decay_rate = config["learning"]["decay_rate"]
    batch_size = config["learning"]["batch_size"]
    refresh_rate = config["display"]["refresh_rate"]

    bar = trange(num_iters)

    for i in bar:
        batch_indices = rng.uniform(
            shape=[batch_size], maxval=num_samples, dtype=tf.int32
        )
        with tf.GradientTape() as tape:
            x_batch = tf.gather(x, batch_indices)
            y_batch = tf.gather(y, batch_indices)

            y_hat = basis(x_batch)
            loss = tf.math.reduce_mean((y_batch - y_hat) ** 2 / 2)

        grads = tape.gradient(loss, basis.trainable_variables)
        grad_update(step_size, basis.trainable_variables, grads)

        step_size *= decay_rate

        if i % refresh_rate == (refresh_rate - 1):
            bar.set_description(
                f"Step {i}; Loss => {loss.numpy():0.4f}, step_size => {step_size:0.4f}"
            )
            bar.refresh()

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(x.numpy().squeeze(), y.numpy().squeeze(), "x")
    a = tf.linspace(tf.reduce_min(x), tf.reduce_max(x), 100)[:, tf.newaxis]
    ax1.plot(a.numpy().squeeze(), basis(a).numpy().squeeze(), linestyle="dashed")
    sinx = tf.linspace(0, 1, 100)
    siny = tf.math.sin(2 * 3.14159 * sinx)
    ax1.plot(sinx.numpy().squeeze(), siny.numpy().squeeze())
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Fig1")
    h = ax1.set_ylabel("y", labelpad=10)
    h.set_rotation(0)

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Fig1")
    for i in range(0, M):
        ax2.plot(a.numpy().squeeze(), basis.sliver_call(i, a).numpy().squeeze())

    fig.savefig("plot.pdf")
