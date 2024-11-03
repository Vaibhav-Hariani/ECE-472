##This is layernorm if Groups = 1
class GroupNorm(tf.Module):
    def __init__(self, groups, channels, eps=1e-5, seed=42):
        rng = tf.random.get_global_generator()
        rng.reset_from_seed(seed)
        stddev = 1 / channels
        self.gamma = tf.Variable(
            rng.normal(shape=[1, 1, 1, channels], stddev=stddev),
            trainable=True,
            name="GN/gamma",
        )
        self.beta = tf.Variable(
            rng.normal(shape=[1, 1, 1, channels], stddev=stddev),
            trainable=True,
            name="GN/beta",
        )
        self.eps = eps
        self.groups = groups

    # Directly from the paper
    def __call__(self, x):
        N, H, W, C = x.shape
        # C = channels
        x = tf.reshape(x, [N, self.groups, C // self.groups, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta
