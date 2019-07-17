import math

from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
from keras.layers.core import RepeatVector, Lambda
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.models import Model
import nltk
import numpy as np
from keras.optimizers import SGD, Adagrad, Adam, RMSprop
from matplotlib import pyplot

import prepare_data
from keras.models import load_model
import pandas as pd

sent_lens, parsed_sentences, word_freqs = prepare_data.clean_and_tokenize(False)

VOCAB_SIZE = 2000
SEQUENCE_LEN = 40

word2id = {}
word2id["PAD"] = 0
word2id["UNK"] = 1

for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2id[k] = v + 2
id2word = {v: k for k, v in word2id.items()}

# in order to present each word we use Glove word embeddings instead of one-hot-vector, because 1-hot will make it large
# Glove is 50 dim, DATA_DIR: glove folder

# EMBED_SIZE = 50

def lookup_word2id(word):
    try:
        return word2id[word]
    except KeyError:
        return word2id["UNK"]



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
# this code return the id of the sentence for the common words in each sequence

sent_lens = np.array(sent_lens)
print("number of sentences: {:d}".format(len(sent_lens)))
print("distribution of sentence lengths (number of words)")
print("min:{:d}, max:{:d}, mean:{:.3f}, 25quart:{:.3f}, med:{:.3f}, 75quart:{:.3f}".format(np.count_nonzero(np.min(sent_lens)), np.max(sent_lens),
                                                           np.mean(sent_lens),np.percentile(sent_lens,25), np.median(sent_lens), np.percentile(sent_lens, 75)))
print("vocab size (full): {:d}".format(len(word_freqs)))


from keras.utils import to_categorical
def sentence_generator(X, batch_size):
    while True:
        # loop once per epoch
        num_recs = X.shape[0]
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size : (bid + 1) * batch_size]
            temp_sents = X[sids, :]
            Xbatch =to_categorical(temp_sents, num_classes=VOCAB_SIZE, dtype='int32')
            yield Xbatch, Xbatch

LATENT_SIZE = 20

def score_cooccurance(tf_a1):
    import tensorflow as tf
    co_occurance_threshold = 2
    exclude_common_words = 1000
    N = 50

    p = (tf_a1 + tf.abs(tf_a1)) / 2
    input_tf = tf.concat([p, tf.zeros((1, p.shape[1]), p.dtype)], axis=0)
    tf_a2 = tf.sort(sent_wids, axis=1)
    first_col_change = tf.zeros([tf_a2.shape[0], 1], dtype=tf.int32)
    last_cols_change = tf.cast(tf.equal(tf_a2[:, 1:], tf_a2[:, :-1]), tf.int32)
    change_bool = tf.concat([first_col_change, last_cols_change], axis=-1)
    not_change_bool = 1 - change_bool
    tf_a2_changed = tf_a2 * not_change_bool + change_bool * N #here

    #this part is only for selecting word indexes which are not very common
    y, idx, count = tf.unique_with_counts(tf.reshape(tf_a2_changed, [-1, ]))
    count_mask = tf.reshape(tf.gather(count, idx), tf_a2_changed.shape)
    result = tf.where(tf.logical_and(tf.less(count_mask, exclude_common_words),
                                     tf.not_equal(tf_a2_changed, N)),
                      tf_a2_changed,
                      tf.math.negative(tf.ones_like(tf_a2_changed)))

    idx = tf.where(tf.count_nonzero(tf.gather(input_tf, result, axis=0), axis=1) >= co_occurance_threshold)
    y, x = idx[:, 0], idx[:, 1]
    rows_tf = tf.gather(result, y, axis=0)

    columns_tf = tf.cast(x[:, None], tf.int32)

    out = tf.zeros(shape=tf.shape(p), dtype=tf.float32)

    rows_tf = tf.reshape(rows_tf, shape=[-1, 1])

    columns_tf = tf.reshape(
        tf.tile(columns_tf, multiples=[1, tf.shape(result)[1]]),
        shape=[-1, 1])

    sparse_indices = tf.reshape(
        tf.concat([rows_tf, columns_tf], axis=-1),
        shape=[-1, 2])
    v = tf.gather_nd(input_tf, sparse_indices)
    v = tf.reshape(v, [-1, tf.shape(result)[1]])

    p_good_rows = tf.tensor_scatter_update(out, tf.cast(sparse_indices, tf.int32), tf.reshape(v, shape=[-1]))
    p_sum_not_in_goodrows = tf.reduce_sum(p - p_good_rows)
    number_of_good_items = tf.where(p_good_rows)
    enegy = p_sum_not_in_goodrows / tf.cast(tf.shape(number_of_good_items)[0]+1, dtype=tf.float32)
    energy_matrice = tf.scatter_nd(tf.cast(number_of_good_items, tf.int32),
                                   enegy * tf.ones(shape=(tf.shape(number_of_good_items)[0])),
                                   shape=tf.shape(p))
    result_p = p_good_rows + energy_matrice

    ####story for the negative weights

    n = (tf_a1 - tf.abs(tf_a1)) / 2
    input_tf_n = tf.concat([n, tf.zeros((1, n.shape[1]), n.dtype)], axis=0)

    idx_n = tf.where(tf.count_nonzero(tf.gather(input_tf_n, result, axis=0), axis=1) >= co_occurance_threshold)
    y_n, x_n = idx_n[:, 0], idx_n[:, 1]
    rows_tf_n = tf.gather(result, y_n, axis=0)
    columns_tf_n = tf.cast(x_n[:, None], tf.int32)
    out_n = tf.zeros(shape=tf.shape(n), dtype=tf.float32)

    rows_tf_n = tf.reshape(rows_tf_n, shape=[-1, 1])
    columns_tf_n = tf.reshape(
        tf.tile(columns_tf_n, multiples=[1, tf.shape(result)[1]]),
        shape=[-1, 1])
    sparse_indices_n = tf.reshape(
        tf.concat([rows_tf_n, columns_tf_n], axis=-1),
        shape=[-1, 2])
    v_n = tf.gather_nd(input_tf_n, sparse_indices_n)
    v_n = tf.reshape(v_n, [-1, tf.shape(result)[1]])

    n_good_rows = tf.tensor_scatter_update(out_n, tf.cast(sparse_indices_n, tf.int32), tf.reshape(v_n, shape=[-1]))

    n_sum_not_in_goodrows = tf.reduce_sum(n - n_good_rows)
    n_number_of_good_items = tf.where(n_good_rows)
    enegy_n = n_sum_not_in_goodrows / tf.cast(tf.shape(n_number_of_good_items)[0] +1 , dtype=tf.float32)
    energy_matrice_n = tf.scatter_nd(tf.cast(n_number_of_good_items, tf.int32),
                                   enegy_n * tf.ones(shape=(tf.shape(n_number_of_good_items)[0])),
                                   shape=tf.shape(n))
    result_n = n_good_rows + energy_matrice_n
    res = result_p + result_n
    return res

train_size = 0.8
# print(encoded.shape)
# encoded = entropy_measure(beta=8,batch_size=BATCH_SIZE)(encoded)
BATCH_SIZE = 64
split_index = int(math.ceil(len(sent_wids)*train_size))
Xtrain = sent_wids[0:split_index, :]
Xtest = sent_wids[split_index:, :]
train_gen = sentence_generator(Xtrain,  BATCH_SIZE)
test_gen = sentence_generator(Xtest, BATCH_SIZE)
NUM_EPOCHS = 40

num_train_steps = len(Xtrain) // BATCH_SIZE
num_test_steps = len(Xtest) // BATCH_SIZE
folder = '50-bs64-sgd-0.001-seq40'
inputs = Input(shape=(SEQUENCE_LEN, VOCAB_SIZE), name="input")
encoded = GRU(LATENT_SIZE,activation='tanh', kernel_initializer="glorot_normal",name="encoder_lstm")(inputs)
# encoded = Lambda(score_cooccurance,  name='modified_layer')(encoded)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = GRU(VOCAB_SIZE, activation='tanh',return_sequences=True)(decoded)
autoencoder = Model(inputs, decoded)
# sgd = SGD(lr=0.001, momentum=0.)
# sgd = RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
sgd = SGD(lr=0.001, momentum=0.9, clipnorm=1.0, nesterov=True)
# sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
autoencoder.compile(optimizer=sgd, loss='categorical_crossentropy')
autoencoder.summary()
early_stopper =  EarlyStopping(monitor='val_loss', patience=2)
checkpoint = ModelCheckpoint(filepath='checkpoint/'+folder+'/{epoch}.hdf5')
history = autoencoder.fit_generator(train_gen, steps_per_epoch=num_train_steps,validation_data=test_gen, validation_steps=num_test_steps, epochs=NUM_EPOCHS, callbacks=[early_stopper, checkpoint])
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.title('model train vs validation loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'validation'], loc='upper right')
pyplot.show()
train_gen = None
word_freqs = None
parsed_sentences = None
print('111')
from keras import backend as K
K.clear_session()

# tw = np.array([[]])
# def saving_df(topics):
#     for topic in topics:
#         temp_df = pd.DataFrame(topic, columns=['epoch', 'index', 'word', 'weight'])
#         with open('./csvfiles/lstm_ae_100e.csv', 'a') as f:
#             temp_df.to_csv(f, index=False)
import gc
print('222')
gc.collect()
tw = [[]]
# print('start', process.memory_info().rss)
for epoch in range(1, NUM_EPOCHS + 1):
    # df = pd.DataFrame(columns=['epoch', 'index', 'word', 'weight'])
    file_name = 'checkpoint/'+folder+'/" + str(epoch) + ".hdf5'
    # lstm_autoencoder = load_model(file_name, {'sent_wids': sent_wids, 'score_cooccurance': score_cooccurance})
    # encoder = Model(lstm_autoencoder.input, lstm_autoencoder.get_layer('modified_layer').output)
    lstm_autoencoder = load_model(file_name, {'sent_wids': sent_wids})
    encoder = Model(lstm_autoencoder.input, lstm_autoencoder.get_layer('encoder_lstm').output)
    topics = []
    weights = encoder.get_weights()[0]
    print(weights)
    # np.append(tw, weights)
    # np.save('test3_2.npy', weights)
    for idx in range(encoder.output_shape[1]):
        token_idx = np.argsort(weights[:, idx])[::-1]
        topics.append([(epoch, idx, id2word[x], weights[x, idx]) for x in token_idx if x in id2word])

    lstm_autoencoder = None
    for topic in topics:
        temp_df = pd.DataFrame(topic, columns=['epoch', 'index', 'word', 'weight'])
        with open('./csvfiles/lstm_ae_without_layer_50e.csv', 'a') as f:
            temp_df.to_csv(f, index=False)
        # print('df created1', process.memory_info().rss)
        del temp_df
        gc.collect()
        temp_df = pd.DataFrame()
        # print('df created2', process.memory_info().rss)
        topic = None
        # print('df created3', process.memory_info().rss)
    topics = None

    # print('df created4', process.memory_info().rss)

    print(epoch)

