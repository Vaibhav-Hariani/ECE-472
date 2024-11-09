from transformer import *
import tensorflow as tf

class TestEmbeddings():
    Corpus = "Now this is a now This is a story all about how"
    Embeddings = Pre_Processor(Corpus, 1000, 4,transformer_seq_length=100)

    def test_embeddings_size(self):
        ##8 distinct words should be broken up into 
        assert TestEmbeddings.Embeddings.embeddings_size == 10

    def test_positional_encodings(self):
        attempt1 = "Now this is"
        attempt2 = "is this now"
        d1 = TestEmbeddings.Embeddings(attempt1)
        d2 = TestEmbeddings.Embeddings(attempt2)
        ##Confirm that the first element is not the same as the last element
        ##Confirm that all of these tensors are equivalent and not equivalent
        assert(tf.math.reduce_all(tf.equal(d1[1],d2[1])))
        assert(not tf.math.reduce_all(tf.equal(d1[0],d2[3])))
        assert(tf.math.reduce_all(tf.equal(d1[1],d2[1])))

class TestDecoder():
    Corpus = "Now this is a story all about how My life got flipped turned upside down and Id like"
    Embeddings = Pre_Processor(Corpus, 100, 4,transformer_seq_length=20)
    Transformer = Decoder(Embeddings.embeddings_size,20, 5,1)

    def test_jacobian(self):
        embeddings = TestDecoder.Embeddings("This story")
        embeddings = tf.cast(embeddings,tf.float32)
        embeddings = tf.expand_dims(embeddings,0)
        with tf.GradientTape() as tape:
            tape.watch(embeddings)
            predicted = TestDecoder.Transformer(embeddings=embeddings)

        j = tape.batch_jacobian(predicted,embeddings)
        upper_triangular = tf.linalg.band_part(j,0,-1)
        assert(tf.math.reduce_all(tf.equal(j,upper_triangular)))