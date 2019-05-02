#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os

__author__ = 'Jayeol Chun'


# resource paths
HOME = os.path.expanduser("~") # [lycastus] /home/b/jchun
RESOURCES = os.path.join(HOME, "Documents/Resources")
W2V = os.path.join(RESOURCES, "w2v")
BERT = os.path.join(RESOURCES, "bert_models")
BERT_TEMPLATE = os.path.join(BERT, "{}_L-12_H-768_A-12")
BERT_CONFIG_FILE = "bert_config.json"
BERT_VOCAB_FILE = "vocab.txt"
BERT_CKPT_FILE = "bert_model.ckpt"

# data paths
WSJ = "/home/j/clp/chinese/corpora/wsj"
PDTB = "./data/pdtb"
CONLL = os.path.join(PDTB, "conll")

# config choices
MODELS = ['mlp',  # TE's best model
          'inter_attn', 'inter_attention',
          'self_attn', 'self_attention',
          'inter_intra_attn', 'inter_intra_attention',
          'inter_self_attn', 'inter_self_attention']
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
OPTIMIZERS = ['sgd', 'adam', 'adagrad']
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
