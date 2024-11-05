from attention import MultiHeadAttention
import tensorflow as tf
from resnet_linear import ResidualBlock
import numpy as np
from mlp import MLP


class Pre_Processor(tf.Module):
    def __init__(self, corpus, n, d):

        self.n = n
        self.d = d

        cur_words = 2
        self.map_of_words = {}
        self.int_to_words = {}
        
        ##Padding/EOS Tokens
        ##Got these from the tensorflow word2vec reference
        ##Only realized that existed after I'd built the rest of this :)
        self.map_of_words["<pad>"] = 0
        self.int_to_words[0] = "<pad>"
        self.map_of_words["<EOS>"] = 1
        self.int_to_words[1] = "<EOS>"
        ##A first pass attempt at building an embedding space given a corpus
        ##This has the limitation of not being able to parse tokens not in the corpus
        for line in corpus:
            for word in line.lower().split():
                if word not in self.map_of_words:
                    self.map_of_words[word] = cur_words
                    self.int_to_words[cur_words] = word
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
        column_row = np.arange(seq.shape[1], dtype=np.int16)
        ##Using ints to round down to 2
        column_row = column_row / 2
        ##Now we need to get fractions...
        column_row = 2 * column_row.astype(np.float32) / self.d
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
    ##And then does positional encoding.
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

    ##Allows the system to train its embedding space a la word2vec 
    def train_embeddings_custom(self,corpus,context_size):
        ##This came out of tensorflow docs
        pos_skip_grams = []
        neg_skip_grams = []
        for line in corpus:
            words = line.lower().split()
            ##This slowly loops over the entire 
            for i in range(context_size,len(words) - context_size):
                k = 0
                while(k < context_size):
                    ##Each context, target pair can be flipped
                    pos_skip_grams.append(words[i],words[i-k])
                    pos_skip_grams.append(words[i-k],words[i])
                    pos_skip_grams.append(words[i],words[i+k])



class TransformerBlock(tf.Module):
    ##Mask necessary for first block in decoder
    def __init__(self, input_dim, num_heads, dropout_rate=0.2, seed=[0, 42]):
        self.attn = MultiHeadAttention(
            input_dim, input_dim, num_heads, dropout_rate=dropout_rate, seed=seed
        )
        self.ff = ResidualBlock(
            input_dim=input_dim,
            output_dim=input_dim,
            dropout_rate=dropout_rate,
            seed=seed,
        )

    def __call__(self, x, mask=None, dropout=False):
        attention_out = x + self.attn(x, mask=mask, dropout=dropout)
        return self.ff(attention_out, dropout=dropout)


class Decoder(tf.Module):
    def __init__(self, Corpus, transformer_dim, num_blocks, num_heads):
        ##This is pretty standard
        self.embeddings = Pre_Processor(Corpus, 1000, 4)
        embeddings_dim = self.embeddings.embeddings_size
        self.input_layer = MLP(
            embeddings_dim,
            transformer_dim,
            2,
            transformer_dim,
            hidden_activation=tf.nn.relu,
            output_activation=tf.nn.relu,
            dropout_rate=0.2,
        )
        self.transformer_blocks = []
        for i in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(transformer_dim, num_heads))
        self.output_layer = MLP(
            transformer_dim,
            embeddings_dim,
            1,
            hidden_activation=tf.nn.relu,
            output_activation=tf.nn.relu,
            dropout_rate=0.2,
        )

    def __call__(self, raw_seq, mask=None, dropout=False):
        encoded_input = self.embeddings(raw_seq)
        x = self.input_layer(encoded_input, dropout)
        for block in self.transformer_blocks:
            x = block(x, mask, dropout)
        return self.output_layer(x, dropout)
