#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from data import PDTBProcessor
from embedding import Embedding

__author__ = 'Jayeol Chun'


class Experiment(object):
  def __init__(self, hp):
    self.hp = hp

    # init data preprocessor
    self.processor = PDTBProcessor()

    vocab = None
    if not self.hp.embedding:
      # if no embedding is specified, need to collect vocab from training set
      self.processor.compile_vocab_labels()
      vocab = self.processor.vocab
    else:
      self.processor.compile_labels()

    # self.labels = self.processor.labels
    self.label_mapping = self.processor.get_label_mapping()

    # init embedding
    if self.hp.embedding == 'bert':
      raise NotImplementedError()
    else:
      self.embedding = Embedding(embedding=self.hp.embedding,
                                 vocab=vocab)
      # embedding_table = self.embedding.get_embedding_table()


  def run(self):
    pass
