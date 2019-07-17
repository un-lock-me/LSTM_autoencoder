import keras
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
import math
import prepare_data
from concentration_scores import KCompetitive
from ent_layer import entropy_measure
import tensorflow as tf

sent_lens, parsed_sentences, word_freqs = prepare_data.clean_and_tokenize(False)
VOCAB_SIZE = 2000
SEQUENCE_LEN = 100

# the input to LSTM is numeric, so we use a lookup table
# lookup table: word2id and id2word

word2id = {}
word2id["PAD"] = 0
word2id["UNK"] = 1
for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2id[k] = v + 2
id2word = {v: k for k, v in word2id.items()}

# in order to present each word we use Glove word embeddings instead of one-hot-vector, because 1-hot will make it large
# Glove is 50 dim, DATA_DIR: glove folder

EMBED_SIZE = 50

def lookup_word2id(word):
    try:
        return word2id[word]
    except KeyError:
        return word2id["UNK"]

# def load_glove_vectors(glove_file, word2id, embed_size):
#     embedding = np.zeros((len(word2id), embed_size))
#     fglove = open(glove_file, "r")
#     vec = np.zeros(embed_size)
#     for line in fglove:
#         cols = line.strip().split()
#         word = cols[0]
#         # print(word)
#         if embed_size == 0:
#             embed_size = len(cols) - 1
#         if word in word2id:
#             vec = np.array([float(v) for v in cols[1:]])
#         embedding[lookup_word2id(word)] = vec
#     embedding[word2id["PAD"]] = np.zeros((embed_size))
#     embedding[word2id["UNK"]] = np.random.uniform(-1, 1, embed_size)
#     return embedding

# embeddings = load_glove_vectors(os.path.join(
#     'Data/glove.6B/', "glove.6B.{:d}d.txt".format(EMBED_SIZE)), word2id, EMBED_SIZE)
model_wv = Word2Vec.load("word2vec_50d_7w")
embeddings = np.zeros((len(model_wv.wv.vocab), EMBED_SIZE))
for i in range(len(model_wv.wv.vocab)):
    # print(i)
    embedding_vector = model_wv.wv[model_wv.wv.index2word[i]]
    if embedding_vector is not None:
        embeddings[i] = embedding_vector

# print(embeddings)
# generator will shuffle sentences at the beginning of each epoch, returns 64 batdfch sentences
# each sentence is represented with Glove vectors
# two generators are defined, one for train and one for test (70% and 30%)

BATCH_SIZE = 512

sent_wids = np.zeros((len(parsed_sentences),SEQUENCE_LEN),'int32')
for index_sentence in range(len(parsed_sentences)):
    temp_sentence = parsed_sentences[index_sentence]
    temp_words = nltk.word_tokenize(temp_sentence)
    for index_word in range(SEQUENCE_LEN):
        if index_word < sent_lens[index_sentence]:
            sent_wids[index_sentence,index_word] = lookup_word2id(temp_words[index_word])
        else:
            sent_wids[index_sentence, index_word] = lookup_word2id('PAD')

def sentence_generator(X, embeddings, batch_size):
    while True:
        # loop once per epoch
        num_recs = X.shape[0]
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size: (bid + 1) * batch_size]
            temp_sents = X[sids, :]
            Xbatch = embeddings[temp_sents]
            # print(Xbatch.shape)
            yield Xbatch, Xbatch

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
        rev = tf.map_fn(row_entropy, encoded, dtype=tf.float32)
        rev = tf.where(tf.is_nan(rev), tf.zeros_like(rev), rev)
        print(rev.shape)
        rev = tf.cast(rev, tf.float32)
        max_entropy = tf.log(tf.clip_by_value(nw, 2, LATENT_SIZE))
        concentration = (max_entropy/(1+rev))
        new_x = x * (tf.reshape(concentration, [BATCH_SIZE, 1]))
        return new_x

train_size = 0.95
split_index = int(math.ceil(len(sent_wids)*train_size))
Xtrain = sent_wids[0:split_index, :]
Xtest = sent_wids[split_index:, :]
# Xtrain, Xtest = train_test_split(sent_wids, train_size=train_size)
train_gen = sentence_generator(Xtrain, embeddings, BATCH_SIZE)
test_gen = sentence_generator(Xtest, embeddings, BATCH_SIZE)

LATENT_SIZE = 20
NUM_EPOCHS = 100
num_train_steps = len(Xtrain) // BATCH_SIZE
num_test_steps = len(Xtest) // BATCH_SIZE

inputs = Input(shape=(SEQUENCE_LEN, EMBED_SIZE), name="input")
encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum", name="encoder_lstm")(inputs)
# encoded =Lambda(rev_entropy)(encoded)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True), merge_mode="sum", name="decoder_lstm")(decoded)
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer="sgd", loss='mse')
autoencoder.summary()
checkpoint = ModelCheckpoint(filepath='checkpoint/{epoch}.hdf5')
history = autoencoder.fit_generator(train_gen, steps_per_epoch=num_train_steps, epochs=NUM_EPOCHS, validation_data=test_gen, validation_steps=num_test_steps, callbacks=[checkpoint])
from keras.models import load_model
import pandas as pd
encoder_model = Model(inputs,encoded)

df = pd.DataFrame(columns=['epoch', 'index', 'word', 'weight'])

for epoch in range(1, NUM_EPOCHS + 1):
    file_name = "checkpoint/" + str(epoch) + ".hdf5"
    autoencoder = load_model(file_name)
    encoder = Model(autoencoder.input, autoencoder.get_layer('encoder_lstm').output)
    topics = []
    weights = encoder.get_weights()[0]
    for idx in range(encoder.output_shape[1]):
        token_idx = np.argsort(weights[:, idx])[::-1]
        topics.append([(epoch, idx, id2word[x], weights[x, idx]) for x in token_idx if x in id2word])
    for topic in topics:
        temp_df = pd.DataFrame(topic, columns=['epoch', 'index', 'word', 'weight'])
        # df = df.append(temp_df, ignore_index=True)
        with open('dataframe_withoutlayer_lstmae.csv', 'a') as f:
            temp_df.to_csv(f, index=False)
    print(epoch)
# from keras.models import load_model
#
# # model = load_model('./Data/simple_ae_to_compare.hdf5')
# # encoder = Model(model.input,model.get_layer('encoder_lstm').output)

# def get_topics_strength(model, vocab, topn=10):
#     topics = []
#     listx =[]
#     weights = model.get_weights()[0]
#
#     for idx in range(model.output_shape[1]):
#         print(idx)
#         token_idx = np.argsort(weights[:, idx])[::-1][:topn]
#         print(weights)
#         print(token_idx)
#         topics.append([(vocab[x], weights[x, idx]) for x in token_idx if x in vocab])
#     return topics
#
# topics = get_topics_strength(encoder_model, id2word, topn=10)
# print(topics)


