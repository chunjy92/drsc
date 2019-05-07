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
MODELS = ['mlp',  # Te's best model
          'bert', # BERT only
          'inter_attn', 'inter_attention',
          'self_attn', 'self_attention',
          'inter_intra_attn', 'inter_intra_attention',
          'inter_self_attn', 'inter_self_attention']
EMBEDDINGS = ['random_init',
              'bert', # BERT as word embedding only
              'glove.6B.50d', # uncased
              'glove.6B.100d', # uncased
              'glove.6B.200d', # uncased
              'glove.6B.300d', # uncased
              'glove.42B.300d', # uncased
              'glove.840B.300d', # cased
              'googlenews']
TRUNC_MODES = ['normal', 'reverse', 'reverse_arg1', 'reverse_arg2']
POOLING_ACTIONS = ['sum', 'mean', 'max', 'concat', 'matmul',
                   # second line applies to `Attentional` model only
                   'first_cls', 'second_cls', 'new_cls']
CONN_ACTIONS = []
PADDING_ACTIONS = ['normal', 'pad_left_arg1']
OPTIMIZERS = ['sgd', 'adam', 'adagrad']
SENSE_TYPES = ['all', 'implicit', 'non-explicit', 'explicit']
MULTIPLE_SENSES_ACTIONS = ['pick_first', 'duplicate']

# special tokens
PAD = '_PAD_'
UNK = '_UNK_'

# PDTB data.json fields
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

# PDTB parse.json fields
SENTENCES = 'sentences'
WORDS = 'words'

# data.json and parse.json alignment
SENT_ID = -2 # == 3
TOK_ID = -1 # == 4


IMPLICIT_4_WAY = [
  'Comparison',
  'Contingency',
  'Expansion',
  'Temporal'
]

# List of senses
IMPLICIT_11_WAY = [
  # 2 Comparisons
  'Comparison.Concession',
  'Comparison.Contrast',
  # 2 Contingencies
  'Contingency.Cause',
  'Contingency.Condition',
  # 5 Expansions
  'Expansion.Alternative',
  'Expansion.Conjunction',
  'Expansion.Exception',
  'Expansion.Instantiation',
  'Expansion.Restatement',
  # 2 Temporals
  'Temporal.Synchrony',
  'Temporal.Asynchronous'
]

# 11_WAY + first-level senses, sorting to ensure same index-label mapping
IMPLICIT_15_WAY = sorted(IMPLICIT_4_WAY + IMPLICIT_11_WAY)

# from https://github.com/nikitakit/self-attentive-parser/blob/master/src/parse_nk.py
# TODO (May 5): currently not used
BERT_TOKEN_MAPPING = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "``": '"',
    "''": '"',
    "`": "'",
    '«': '"',
    '»': '"',
    '‘': "'",
    '’': "'",
    '“': '"',
    '”': '"',
    '„': '"',
    '‹': "'",
    '›': "'",
}
