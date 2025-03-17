# coding: utf-8
# sequence to sequence network
get_ipython().run_cell_magic('capture', '', '# sequence to sequence network\n# in which two recurrent neural nets work together\n# to transform on sequence to another\n# an encoder network condenses an input seq to a vec\n# and a decoder net unfolds that vec into a new seq\n\n')
get_ipython().run_cell_magic('capture', '', '# this impl will include an attention mechanism\n# whih lets the decoder learn ot focus over a specific\n# range of the input sequence\n\n')
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
get_ipython().run_line_magic('clear', '')
get_ipython().run_cell_magic('capture', '', '# represent each word as a one hot encoding\n# -> a giant vector of zeros except for a single 1\n# the 1 is just the index of the word\n\n')
get_ipython().run_cell_magic('capture', '', '# start and end of seq tokens\nSOS_token = 0\nEOS_token = 1\n\n')
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n        \n')
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n        \n')
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n    \n')
get_ipython().run_line_magic('clear', '')
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n        \n    def addSentence(self, sentence):\n        for word in sentence.split(\' \'):\n            self.adWord(word)\n            \n    def addWord(self, word):\n        if word not in self.word2index:\n            self.word2index[word] = self.num_words\n            self.word2count[word] = 1\n            self.index2word[self.num_words] = word\n            self.num_words += 1\n        else:\n            self.word2count[word] += 1\n                \n')
get_ipython().run_cell_magic('capture', '', "# turn unicode string to ASCII\ndef unicodeToAscii(s):\n    return ''.join(\n    )\n")
get_ipython().run_cell_magic('capture', '', "# turn unicode string to ASCII\ndef unicodeToAscii(s):\n    return ''.join(\n        c for c in unicodedata.normalize('NFD', s)\n        if unicodedata.category(c) != 'Mn'\n    )\n    \n")
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sib(r"([.!?]), r" \1", s)
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sib(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()
    
def readLangs(lang1, lang2, reverse=False):
    # files will be in eglish and some other lang
    # split the file into lines, and split lines into pairs
    print("reading lines...")
    # read the file and split into lines, searching for langs
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    
get_ipython().run_line_magic('clear', '')
def readLangs(lang1, lang2, reverse=False):
    # files will be in eglish and some other lang
    # split the file into lines, and split lines into pairs
    print("reading lines...")
    # read the file and split into lines, searching for langs
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # split lines into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    
def readLangs(lang1, lang2, reverse=False):
    # files will be in eglish and some other lang
    # split the file into lines, and split lines into pairs
    print("reading lines...")
    # read the file and split into lines, searching for langs
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # split lines into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # reverse pairs, make lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input lang = Lang(lang1)
def readLangs(lang1, lang2, reverse=False):
    # files will be in eglish and some other lang
    # split the file into lines, and split lines into pairs
    print("reading lines...")
    # read the file and split into lines, searching for langs
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')
    # split lines into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    # reverse pairs, make lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs
    
get_ipython().run_line_magic('clear', '')
get_ipython().run_cell_magic('capture', '', '# there are many samples, so to train quickly we trim dataset\n# to only short sentences\n# here max length is 10 words\nMAX_LENGTH = 10\neng_prefixes = (\n"i am ", "i m ", "he is ", "he is ", "she is ", "she s ", "you are ", "you re ",\n"we are ", "we re ", "they are ", "they re "\n)\n\n')
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and 
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)
        
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
    
def prepareData(lang1, lang2, reverse=False):
    # read text file, split into lines, split into pairs
    # normalize text, filter by len and content
    # make word lists from sentences in pairs
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("trimmed to %s sentence pairs" % len(pairs))
    print("counting words...")
    for pair in pairs:
        inpuit_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs
    
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
get_ipython().run_line_magic('clear', '')
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()
    
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
def prepareData(lang1, lang2, reverse=False):
    # read text file, split into lines, split into pairs
    # normalize text, filter by len and content
    # make word lists from sentences in pairs
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("trimmed to %s sentence pairs" % len(pairs))
    print("counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs
    
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
get_ipython().run_cell_magic('capture', '', 'class Lang: \n    def __init__(self, name):\n        self.name = name\n        self.word2index = {}\n        self.word2count = {}\n        self.index2word = {0: "SOS", 1: "EOS"}\n        self.num_words = 2 # count SOS and EOS\n        \n    def addSentence(self, sentence):\n        for word in sentence.split(\' \'):\n            self.addWord(word)\n            \n    def addWord(self, word):\n        if word not in self.word2index:\n            self.word2index[word] = self.num_words\n            self.word2count[word] = 1\n            self.index2word[self.num_words] = word\n            self.num_words += 1\n        else:\n            self.word2count[word] += 1\n            \n')
get_ipython().run_line_magic('clear', '')
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
def prepareData(lang1, lang2, reverse=False):
    # read text file, split into lines, split into pairs
    # normalize text, filter by len and content
    # make word lists from sentences in pairs
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("trimmed to %s sentence pairs" % len(pairs))
    print("counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("counted words...")
    print(input_lang.name, input_lang.num_words)
    print(output_lang.name, output_lang.num_words)
    return input_lang, output_lang, pairs
    
get_ipython().run_line_magic('clear', '')
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
