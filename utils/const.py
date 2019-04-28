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
              'glove.6B.50d',
              'glove.6B.100d',
              'glove.6B.200d',
              'glove.6B.300d',
              'glove.840B.300d',
              'googlenews']

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
