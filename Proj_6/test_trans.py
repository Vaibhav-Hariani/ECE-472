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
        ##Verify that tensors are all of expected shape
        assert(d1.shape[1] == 100)

class TestDecoder():
    Corpus = "Now this is a story all about how My life got flipped turned upside down and Id like"
    Embeddings = Pre_Processor(Corpus, 1000, 4,transformer_seq_length=100)
    Transformer = Decoder(Embeddings.embeddings_size,1,5,Embeddings)

    def test_jacobian(self):
        Transformer = TestDecoder.Transformer
        embeddings = TestDecoder.Embeddings("This story")
        with tf.GradientTape() as tape:
            predicted = Transformer(embeddings=embeddings)
        j = tape.jacobian(embeddings,predicted)
        print(j)
