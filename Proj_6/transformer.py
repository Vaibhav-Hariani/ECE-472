import numpy as np
import tensorflow as tf
from attention import MultiHeadAttention
from einops import rearrange
from groupnorm import GroupNorm
from mlp import MLP


class Pre_Processor(tf.Module):
    def __init__(self, corpus, n, d, transformer_seq_length):

        self.n = n
        self.d = d

        self.output_len = transformer_seq_length
        cur_words = 2
        self.map_of_words = {}
        self.int_to_words = {}

        ##Padding/EOS Tokens
        ##Got these from the tensorflow word2vec reference
        ##Only realized that existed after I'd built the rest of this :)
        self.map_of_words["<PAD>"] = 0
        self.int_to_words[0] = "<PAD>"
        self.map_of_words["<EOS>"] = 1
        self.int_to_words[1] = "<EOS>"
        ##A first pass attempt at building an embedding space given a corpus
        ##This has the limitation of not being able to parse tokens not in the corpus
        for word in corpus.lower().split():
            if word not in self.map_of_words:
                self.map_of_words[word] = cur_words
                self.int_to_words[cur_words] = word
                cur_words += 1
        self.embeddings_size = cur_words

    ##This is cursed to read
    ##But Numpy is so powerful
    def positional(self, seq):
        ##The number of elements in the sequence
        ##Assuming it's formatting as nxE: n is number of elements in sequence
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
        tokens.append("<EOS>")
        embeddings = np.zeros((self.embeddings_size, self.output_len))
        for token_i in range(len(tokens)):
            if tokens[token_i] in self.map_of_words:
                index = self.map_of_words[tokens[token_i]]
                embeddings[token_i, index] = 1
            ##dim T x E, T is sequencel length
        return tf.transpose(self.positional(embeddings))

    # ##Allows the system to train its embedding space a la word2vec
    # def train_embeddings_custom(self,corpus,context_size):
    #     ##This came out of tensorflow docs
    #     pos_skip_grams = []
    #     neg_skip_grams = []
    #     for line in corpus:
    #         words = line.lower().split()
    #         ##This slowly loops over the entire
    #         for i in range(context_size,len(words) - context_size):
    #             k = 0
    #             while(k < context_size):
    #                 ##Each context, target pair can be flipped
    #                 pos_skip_grams.append(words[i],words[i-k])
    #                 pos_skip_grams.append(words[i-k],words[i])
    #                 pos_skip_grams.append(words[i],words[i+k])


class TransformerBlock(tf.Module):
    ##Mask necessary for first block in decoder
    def __init__(self, input_dim, embed_dim, num_heads, dropout_rate=0.2, seed=[0, 42]):
        self.attn = MultiHeadAttention(
            input_dim, embed_dim, num_heads, dropout_rate=dropout_rate, seed=seed
        )
        self.layer_1 = GroupNorm(1, input_dim)

        self.FF = MLP(input_dim, input_dim, 2, 15)
        self.layer_1 = GroupNorm(1, input_dim)
        self.layer_2 = GroupNorm(1, input_dim)

    def __call__(self, x, mask=None, dropout=False):
        attention_out = x + self.attn(x, mask=mask, dropout=dropout)
        attention_out = tf.expand_dims(attention_out, 0)
        layer1 = self.layer_1(attention_out)
        linear_out = self.FF(layer1)
        return self.layer_2(linear_out)


class Decoder(tf.Module):
    def __init__(
        self,
        transformer_dim,
        embed_dim,
        num_heads,
        num_blocks,
    ):
        ##Allows the embeddor to be generated ahead of time
        ## (And potentially trained)
        # self.embeddings= embeddor
        # if (not self.embeddings):
        #     self.embeddings = Pre_Processor(Corpus, 1000,4,transformer_dim)
        self.transformer_blocks = []
        for i in range(num_blocks):
            self.transformer_blocks.append(
                TransformerBlock(transformer_dim, embed_dim, num_heads)
            )

    def __call__(self, embeddings, dropout=False):
        ##Get the input, and then mask it into an upper triangular matrix:

        ##No need for a mask into attention as it's built into the call here
        current = tf.linalg.band_part(embeddings, 0, -1)

        # mask = tf.ones_like(embeddings)
        # current = self.embeddings(raw_seq)
        for block in self.transformer_blocks:
            current = block(current, dropout)
        return current


# if __name__ == "__main__":
#     Corpus = "Now this is a story all about how My life got flipped turned upside down and Id like"
#     Embeddings = Pre_Processor(Corpus, 100, 4, transformer_seq_length=20)
#     Transformer = Decoder(Embeddings.embeddings_size, 20, 5, 1)
