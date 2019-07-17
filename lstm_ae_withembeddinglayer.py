import keras
# from gensim.models import Word2Vec
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Flatten, Activation, add, TimeDistributed, Embedding
from keras.layers.core import RepeatVector,Permute,Lambda
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import nltk
import numpy as np
import os
import prepare_data
import tensorflow as tf


sent_lens, parsed_sentences, word_freqs = prepare_data.clean_and_tokenize(False)
VOCAB_SIZE = 2000
SEQUENCE_LEN = 200

word2id = {}
word2id["PAD"] = 0
word2id["UNK"] = 1

for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2id[k] = v + 2
id2word = {v:k for k, v in word2id.items()}

# in order to present each word we use Glove word embeddings instead of one-hot-vector, because 1-hot will make it large
# Glove is 50 dim, DATA_DIR: glove folder

EMBED_SIZE = 50

def lookup_word2id(word):
    try:
        return word2id[word]
    except KeyError:
        return word2id["UNK"]

BATCH_SIZE = 512

model_wv = Word2Vec.load("word2vec_50d_7w")
embeddings = np.zeros((len(model_wv.wv.vocab), EMBED_SIZE))
for i in range(len(model_wv.wv.vocab)):
    # print(i)
    embedding_vector = model_wv.wv[model_wv.wv.index2word[i]]
    if embedding_vector is not None:
        embeddings[i] = embedding_vector

sent_wids = np.zeros((len(parsed_sentences),SEQUENCE_LEN),'int32')
sample_seq_weights = np.zeros((len(parsed_sentences),SEQUENCE_LEN),'float')
for index_sentence in range(len(parsed_sentences)):
    temp_sentence = parsed_sentences[index_sentence]
    temp_words = nltk.word_tokenize(temp_sentence)
    for index_word in range(SEQUENCE_LEN):
        if index_word < sent_lens[index_sentence]:
            sent_wids[index_sentence,index_word] = lookup_word2id(temp_words[index_word])
        else:
            sent_wids[index_sentence, index_word] = lookup_word2id('PAD')
print(sent_wids.shape)
print(sent_wids)

def sentence_generator(X, embeddings, batch_size):
    while True:
        # loop once per epoch
        num_recs = X.shape[0]
        embed_size = embeddings.shape[1]
        indices = np.random.permutation(np.arange(num_recs))
        # print(embeddings.shape)
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size : (bid + 1) * batch_size]
            # Xbatch is a [batch_size, seq_length] array
            Xbatch = X[sids, :]

            # Creating the Y targets
            Xembed = embeddings[Xbatch.reshape(-1), :]
            # Ybatch will be [batch_size, seq_length, embed_size] array
            Ybatch = Xembed.reshape(batch_size, -1, embed_size)
            yield Xbatch, Ybatch

LATENT_SIZE = 60

def rev_entropy(x):
        def row_entropy(row):
            _, _, count = tf.unique_with_counts(row)
            count = tf.cast(count,tf.float32)
            prob = count / tf.reduce_sum(count)
            prob = tf.cast(prob,tf.float32)
            rev = -tf.reduce_sum(prob * tf.log(prob))
            return rev

        # value_ranges = [-10.0, 100.0]
        # nbins = 50
        # new_f_w_t = tf.histogram_fixed_width_bins(x, value_ranges, nbins)
        nw = tf.reduce_sum(x,axis=1)
        rev = tf.map_fn(row_entropy, x)
        rev = tf.where(tf.is_nan(rev), tf.zeros_like(rev), rev)
        rev = tf.cast(rev, tf.float32)
        max_entropy = tf.log(tf.clip_by_value(nw,2,LATENT_SIZE))
        concentration = (max_entropy/(1+rev))
        new_x = x * (tf.reshape(concentration, [BATCH_SIZE, 1]))
        # print(new_x.shape)
        return new_x

train_size = 0.95

split_index = int(np.math.ceil(len(sent_wids) * train_size))
Xtrain = sent_wids[0:split_index, :]
Xtest = sent_wids[split_index:, :]
# train_w = sample_seq_weights[0: split_index, :]
# test_w = sample_seq_weights[split_index:, :]
train_gen = sentence_generator(Xtrain,embeddings, BATCH_SIZE)
# test_gen = sentence_generator(Xtest ,embeddings, BATCH_SIZE)
NUM_EPOCHS = 1

num_train_steps = len(Xtrain) // BATCH_SIZE
num_test_steps = len(Xtest) // BATCH_SIZE

inputs = Input(shape=(SEQUENCE_LEN, ), name="input")
embedding = Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_SIZE, input_length=SEQUENCE_LEN,trainable=False)(inputs)
encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum", name="encoder_lstm")(embedding)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = LSTM(EMBED_SIZE, return_sequences=True)(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer="sgd", loss='categorical_crossentropy')
autoencoder.summary()

checkpoint = ModelCheckpoint(filepath=os.path.join('Data/', "simple_ae_to_compare"))
history = autoencoder.fit_generator(train_gen, steps_per_epoch=num_train_steps, epochs=NUM_EPOCHS,  validation_steps=num_test_steps)

from keras.models import load_model

import keras.backend as K
encoder_model = Model(inputs, encoded)

# model = load_model('./Data/simple_ae_to_compare.hdf5')
# encoder = Model(model.input,model.get_layer('encoder_lstm').output)

# def get_topics_strength(model, vocab, topn=10):
#     topics = []
#     listx =[]
#     weights = model.get_weights()[0]
#     for idx in range(model.output_shape[1]):
#         token_idx = np.argsort(weights[:, idx])[::-1][:topn]
#         print(token_idx)
#         # topics.append([(vocab[x], weights[x, idx]) for x in token_idx if x in vocab])
#     return topics
#
# topics = get_topics_strength(encoder_model, id2word, topn=10)
# print(topics)


