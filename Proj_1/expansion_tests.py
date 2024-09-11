import pytest


def test_additivity():
    import tensorflow as tf

    from Proj_1.basis_expansion import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[1, num_inputs])

    tf.debugging.assert_near(linear(a + b), linear(a) + linear(b), summarize=2)


def test_homogeneity():
    import tensorflow as tf

    from Proj_1.basis_expansion import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1
    num_test_cases = 100

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    b = rng.normal(shape=[num_test_cases, 1])

    tf.debugging.assert_near(linear(a * b), linear(a) * b, summarize=2)


@pytest.mark.parametrize("num_outputs", [1, 16, 128])
def test_dimensionality(num_outputs):
    import tensorflow as tf

    from Proj_1.basis_expansion import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10

    linear = Linear(num_inputs, num_outputs)

    a = rng.normal(shape=[1, num_inputs])
    z = linear(a)

    tf.assert_equal(tf.shape(z)[-1], num_outputs)


@pytest.mark.parametrize("bias", [True, False])
def test_trainable(bias):
    import tensorflow as tf

    from Proj_1.basis_expansion import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs = 10
    num_outputs = 1

    linear = Linear(num_inputs, num_outputs, bias=bias)

    a = rng.normal(shape=[1, num_inputs])

    with tf.GradientTape() as tape:
        z = linear(a)
        loss = tf.math.reduce_mean(z**2)

    grads = tape.gradient(loss, linear.trainable_variables)

    for grad, var in zip(grads, linear.trainable_variables):
        tf.debugging.check_numerics(grad, message=f"{var.name}: ")
        tf.debugging.assert_greater(tf.math.abs(grad), 0.0)

    assert len(grads) == len(linear.trainable_variables)

    if bias:
        assert len(grads) == 2
    else:
        assert len(grads) == 1


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [([1000, 1000], [100, 100]), ([1000, 100], [100, 100]), ([100, 1000], [100, 100])],
)
def test_init_properties(a_shape, b_shape):
    import tensorflow as tf

    from Proj_1.basis_expansion import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    num_inputs_a, num_outputs_a = a_shape
    num_inputs_b, num_outputs_b = b_shape

    linear_a = Linear(num_inputs_a, num_outputs_a, bias=False)
    linear_b = Linear(num_inputs_b, num_outputs_b, bias=False)

    std_a = tf.math.reduce_std(linear_a.w)
    std_b = tf.math.reduce_std(linear_b.w)

    tf.debugging.assert_less(std_a, std_b)


def test_bias():
    import tensorflow as tf

    from Proj_1.basis_expansion import Linear

    rng = tf.random.get_global_generator()
    rng.reset_from_seed(2384230948)

    linear_with_bias = Linear(1, 1, bias=True)
    assert hasattr(linear_with_bias, "b")

    linear_with_bias = Linear(1, 1, bias=False)
    assert not hasattr(linear_with_bias, "b")
