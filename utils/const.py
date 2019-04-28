#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os

__author__ = 'Jayeol Chun'


# resource paths
HOME = os.path.expanduser("~") # [lycastus] /home/b/jchun
RESOURCES = os.path.join(HOME, "Documents/Resources")
BERT = os.path.join(RESOURCES, "bert_models")
W2V = os.path.join(RESOURCES, "w2v")

# data paths
WSJ = "/home/j/clp/chinese/corpora/wsj"
PDTB = "./data/pdtb"
CONLL = os.path.join(PDTB, "conll")

# config choices
MODELS = ['mlp']
EMBEDDINGS = ['bert',
              'glove.6B.50d', # uncased
              'glove.6B.100d', # uncased
              'glove.6B.200d', # uncased
              'glove.6B.300d', # uncased
              'glove.42B.300d', # uncased
              'glove.840B.300d', # cased
              'googlenews']
TRUNC_MODES = ['normal', 'reverse']
POOLING_ACTIONS = ['sum', 'mean', 'max', 'concat', 'matmul'] # concat?
CONN_ACTIONS = []
OPTIMIZERS = ['sgd', 'adam']
SENSE_TYPES = ['all', 'implicit', 'non-explicit', 'explicit']
MULTIPLE_SENSES_ACTIONS = ['pick_first', 'duplicate']

# special tokens
PAD = '_PAD_'
UNK = '_UNK_'

# pdtb data fields
ARG1 = 'Arg1'
ARG2 = 'Arg2'
CONN = 'Connective'
DOC_ID = 'DocID'
REL_ID = 'ID'
TOKEN_LIST = 'TokenList'
CHAR_SPAN_LIST = 'CharacterSpanList'
RAW_TEXT = 'RawText'
TYPE = 'Type'
SENSE = 'Sense'

# parse json fields
SENTENCES = 'sentences'
WORDS = 'words'

# data.json and parse.json alignment
SENT_ID = -2 # == 3
TOK_ID = -1 # == 4
