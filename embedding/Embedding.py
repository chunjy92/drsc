#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tarfile
import zipfile

import numpy as np

from data import InputFeatures
from utils import const

__author__ = 'Jayeol Chun'


class Embedding(object):
  """Embedding class for:

    (1) randomly initialized embedding table built from  vocabs found in
    training set
    (2) pre-defined static word vectors produced from other W2V models.
    Currently supports: Glove, GoogleNews

  """
  def __init__(self,
               embedding=None,
               vocab=None,
               feature_size=50,
               max_arg_length=128):
    self.embedding = embedding
    self.vocab = vocab

    if not self.embedding and not self.vocab:
      raise ValueError(
        "At least one of `embedding` or `vocab` must be specified.")
    # when both are specified, `embedding_source` takes precedence

    # width of word vector: this will be overwritten if loading from source
    if not self.embedding:
      self.feature_size = feature_size
    else:
      if self.embedding == "googlenews":
        self.feature_size = 300
      else:
        last_split = self.embedding.split(".")[-1]
        self.feature_size = int(last_split[:-1])

    self.max_arg_length = max_arg_length

    self._init_embedding()

  @property
  def embedding_table(self):
    return self._embedding_table

  def _init_embedding(self):
    if not self.embedding:
      # [_PAD_, _UNK_, ...]
      self.vocab.insert(0, const.UNK)
      self.vocab.insert(0, const.PAD)

      vocab_size = len(self.vocab)

      embedding = np.random.standard_normal([vocab_size, self.feature_size])
      embedding[0] = 0 # 0 for padding
    else:
      if self.embedding.startswith("glove"):
        pass
      else:
        pass

    self._embedding_table = embedding

  def get_embedding_table(self):
    return self.embedding_table

  def convert_to_ids(self, examples, label_mapping):
    """

    Args:
      examples: List of `PDTBRelation` instances

    Returns:

    """
    def convert_single_example(example):
      data = []
      for tokens in [example.arg1, example.arg2, example.conn]:
        token_ids = []
        if tokens:
          for token in tokens:
            token_ids.append(
              self.vocab.index(token) if token in self.vocab else 1) # UNK at 1
        else:
          token_ids = [0] * self.max_arg_length # PAD at 0
        data.append(token_ids)

      feature = InputFeatures(arg1=data[0], arg2=data[1], conn=data[2],
                              label=label_mapping[example.label])
      return feature

    features = []
    for example in examples:
      features.append(convert_single_example(example))

    return features

  def convert_to_values(self, examples):
    raise NotImplementedError()
