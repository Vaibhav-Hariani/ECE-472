import tensorflow as tf
from datasets import load_dataset


class Positional_Encoder(tf.Module):
    pass


class Input_Embedding(tf.Module):
    pass


class Output_Embedding(tf.Module):
    pass


class Multi_Headed_Attention(tf.Module):
    def __init__(self, h, d_k, d_v):
        pass

    def __call__(Q: tf.Tensor, K: tf.Tensor, V: tf.Tensor):
        interim = Q @ tf.transpose(K)
        d_k = K.shape()[1]
        interim = tf.nn.softmax(interim / tf.math.sqrt(d_k))
        return interim @ V


class Feed_Forward(tf.Module):
    pass


class ADD_NORM(tf.Module):
    pass


class Encoder(tf.Module):
    pass


class Decoder(tf.Module):
    pass


class Transformer(tf.Module):
    pass


if __name__ == "__main__":
    ##Just get training data
    ds = load_dataset("fancyzhx/ag_news", split="train").with_format("tf")
