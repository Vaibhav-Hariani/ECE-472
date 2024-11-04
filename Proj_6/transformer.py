from attention import MultiHeadAttention
import tensorflow as tf
from resnet_linear import ResidualBlock
import numpy as np

class Pre_Processor(tf.Module):
    def __init__(self, corpus,n,d):
        
        self.n = n
        self.d = d

        cur_words = 0
        self.map_of_words = {}
        ##A first pass attempt at building an embedding space given a corpus
        ##This has the limitation of not being able to parse tokens not in the corpus
        for line in corpus:
            for word in str.split(line):
                if word not in self.map_of_words:
                    self.map_of_words[word] = cur_words
                    cur_words += 1

        self.embeddings_size = cur_words
    

    ##This is cursed to read
    ##But Numpy is so powerful
    def positional(self, seq):
        ##The number of elements in the sequence
        ##Assuming it's formatting as nxE: n is number of sentences
        ## X is encoding space

        index_row = np.arange(seq.shape[0])
        ##For length of embedding space
        column_row = np.arange(seq.shape[1],dtype=np.int16)        
        ##Using ints to round down to 2
        column_row = column_row / 2
        ##Now we need to get fractions...
        column_row = column_row.astype(np.float32) / self.d
        denominator = np.power(self.n, column_row)

        ##By transposing, should get a casted rectangular array
        term = np.atleast_2d(index_row).T / denominator

        ##Sin all even terms 
        term[0::2,] = np.sin(term[0::2,])
        ##Cos all odd terms 
        term[1::2,] = np.cos(term[1::2,])
        return term

    ##This takes in an input string, tokenizes it
    ##Places them in an embedding space,
    ##And then handles positional encoding.
    def __call__(self, context):

        ##Given context: Break it up into tokens     
        tokens = str.split(context)
        
        embeddings = np.zeros(self.embeddings_size)

        ##Assuming that we've seen almost everything        
        for token in tokens:
            if token in self.map_of_words:
                index = self.map_of_words[token]
                embeddings[index] = 1

        return self.positional(embeddings)



class TransformerBlock(tf.Module):
    ##Mask necessary for first block in decoder
    def __init__(self, input_dim, num_heads,dropout_rate=0.2, mask=None, seed = [0,42]):
        self.mask = mask
        self.attn = MultiHeadAttention(input_dim, input_dim, num_heads,dropout_rate=dropout_rate,seed=seed)
        self.ff = ResidualBlock(input_dim=input_dim,output_dim=input_dim,dropout_rate=dropout_rate,seed=seed)

    def call(self, x,dropout=False):
        attention_out = x + self.attn(x,mask=self.mask,dropout=dropout)
        return self.ff(attention_out,dropout=dropout)