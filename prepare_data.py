import os
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import collections
from nltk.stem.porter import PorterStemmer as EnglishStemmer
import string

def vecnorm(vec, norm, epsilon=1e-3):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    """
    if norm not in ('prob', 'max1', 'logmax1'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms include 'prob',\
             'max1' and 'logmax1'." % norm)

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if norm == 'prob':
            veclen = np.sum(np.abs(vec)) + epsilon * len(vec) # smoothing
        elif norm == 'max1':
            veclen = np.max(vec) + epsilon
        elif norm == 'logmax1':
            vec = np.log10(1. + vec)
            veclen = np.max(vec) + epsilon
        if veclen > 0.0:
            return (vec + epsilon) / veclen
        else:
            return vec
    else:
        raise ValueError('vec should be ndarray, found: %s' % type(vec))


paragraph_lists = [[], [], []]

def getlistoffiles(dirname):
    listoffiles = os.listdir(dirname)
    allfiles = list()
    for enty in listoffiles:
        fullpath = os.path.join(dirname,enty)

        if os.path.isdir(fullpath):
            allfiles = allfiles+getlistoffiles(fullpath)
        else:
            allfiles.append(fullpath)
    return allfiles

stop_words = set(stopwords.words('english'))

def readingfiles():
    paras = []
    splitLen = 10
    dirname = './Data/20news/'
    # dirname = './Data/foldertesti/'
    documents = getlistoffiles(dirname)
    for file in documents:
        with open(file, encoding='latin1') as f:
            input = f.read().split('\n')
            at = 1
            for lines in range(0, len(input), splitLen):
                # First, get the list slice
                outputData = input[lines:lines + splitLen]
                outputData = [x.replace('\n','').replace('\t','') for x in outputData if x]
                at += 1
                if len(outputData)>0:
                    paras.append(outputData)

    return paras

def is_number(n):
    temp = nltk.re.sub("[.,-/]", "", n)
    return temp.isdigit()

def clean_and_tokenize(stem):
    para = readingfiles()
    index = np.arange(len(para))
    np.random.shuffle(index)
    new_sents = []
    for i in index.tolist():
        new_sents.append(para[i])
    sents = new_sents
    word_freqs = collections.Counter()
    sent_lens = []
    parsed_sentences = []
    # lst_punctuation = str.maketrans('', '', string.punctuation)
    for sent in sents:

        sent = re.sub('[^a-zA-Z]',' ',str(sent))
        sent = re.sub(' +', ' ', sent)
        parsed_words = []
        # print(sent)
        # print(len(sent.strip(' ')))
        for word in nltk.word_tokenize(re.sub('[%s]' % re.escape(string.punctuation), '', sent)):
            if ~is_number(word) and word.strip().lower() not in stop_words and word.isalpha() and len(word)>2:
                if stem:
                    try:
                        w = EnglishStemmer().stem(word)
                        if w in stop_words:
                            w = ''
                            w = re.sub(' +', '', w)
                    except Exception as e:
                        w = word.strip().lower()
                else:
                    w = word.strip().lower()
                word_freqs[w] += 1
                parsed_words.append(w)
                # print(len(parsed_words))
                # print(parsed_words)
        if len(parsed_words) > 3:
            sent_lens.append(len(parsed_words))
            parsed_sentences.append(" ".join(parsed_words))
        # parsed_sentences.append(parsed_words)
    return sent_lens, parsed_sentences,word_freqs










