from gensim.models import Word2Vec
from nltk.corpus import brown
import prepare_data
# #
sent_lens, parsed_sentences, word_freqs = prepare_data.clean_and_tokenize(False)
# print(len(parsed_sentences))
# print(parsed_sentences[0])
VOCAB_SIZE = 20000
SEQUENCE_LEN = 45

# the input to LSTM is numeric, so we use a lookup table
# lookup table: word2id and id2word

word2id = {}
word2id["PAD"] = 0
word2id["UNK"] = 1
for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2id[k] = v + 2
    # print(v)
    # print(k)
id2word = {v:k for k, v in word2id.items()}
print(len(id2word))
# in order to present each word we use Glove word embeddings instead of one-hot-vector, because 1-hot will make it large
# Glove is 50 dim, DATA_DIR: glove folder

def lookup_word2id(word):
    try:
        return word2id[word]
    except KeyError:
        return word2id["UNK"]
emb_dim = 50

## sentences = [['the','apple','was','delicious'],['Aug','submit','my','paper']]
w2vec = Word2Vec(parsed_sentences, size=emb_dim, window=10, min_count=5,iter=15)
print(len(w2vec.wv.vocab))
w2vec.save('word2vec_50d_7w')
import numpy as np
model_wv = Word2Vec.load("word2vec_50d_7w")
embeddings = np.zeros((len(model_wv.wv.vocab), emb_dim))
for i in range(len(model_wv.wv.vocab)):
    # print(i)
    embedding_vector = model_wv.wv[model_wv.wv.index2word[i]]
    if embedding_vector is not None:
        embeddings[i] = embedding_vector

print(embeddings[[1,2]])
# vector = model_wv.most_similar(['computer'])
# embeddings = np.zeros((len(model_wv.wv.vocab), 50))
# print(len(model_wv.wv.vocab))
# # for i in range(len(model_wv.wv.vocab)):
# #     print (i)
# #     embedding_vector = model_wv.wv[model_wv.wv.index2word[i]]
# #     print (embedding_vector)
# #     if embedding_vector is not None:
# #         embeddings[i] = embedding_vector
#
# print(embeddings.shape)