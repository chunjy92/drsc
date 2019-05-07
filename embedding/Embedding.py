#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

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
               word_vector_width=200,
               max_arg_length=128):
    self.embedding = embedding
    self.vocab = vocab

    # width of word vector: this will be overwritten if loading from source
    if self.embedding == "random_init":
      if not self.vocab:
        raise ValueError(
          "`vocab` must be specifined in advance for `random_init` embedding")

      self.word_vector_width = word_vector_width
    else:
      if self.embedding == "googlenews":
        self.word_vector_width = 300
      else:
        last_split = self.embedding.split(".")[-1]
        self.word_vector_width = int(last_split[:-1])

      if self.word_vector_width != word_vector_width:
        tf.logging.info(
          f"word_vector_width overwritten to {self.word_vector_width}")

    self.max_arg_length = max_arg_length

    self._init_embedding()

  @property
  def embedding_table(self):
    return self._embedding_table

  def _init_embedding(self):
    if self.embedding == 'random_init':
      # [_PAD_, _UNK_, ...]
      self.vocab.insert(0, const.UNK)
      self.vocab.insert(0, const.PAD)

      vocab_size = len(self.vocab)

      embedding = \
        np.random.standard_normal([vocab_size, self.word_vector_width])
      embedding[0] = 0 # 0 for padding
    else:
      assert self.vocab is not None, "requires a reduced number of vocabs"
      if self.embedding.startswith("glove"):

        import zipfile

        if '6B' in self.embedding:
          glove_zip = os.path.join(const.W2V, "glove.6B.zip")
          glove_txt = f"glove.6B.{self.word_vector_width}d.txt"
        else:
          glove_zip = os.path.join(const.W2V, self.embedding + ".zip")
          glove_txt = self.embedding + ".txt"

        archive = zipfile.ZipFile(glove_zip, 'r')
        txt_file = archive.open(glove_txt)

        tf.logging.info(f"Loading {glove_txt}")

        vocab = []
        embedding = []
        for line in tqdm(txt_file.readlines()):
          line = line.decode('utf-8').strip().split()

          word_list = line[:-self.word_vector_width]
          if len(word_list) > 1:
            # for some reason multiple string tokens may exist in a single data
            # instance. For example, [. . . ...] or [at name@domain ...]. We
            # skip these
            continue

          # word = "".join(line[:-self.word_vector_width])
          word = line[0]

          # only collect words that are in the PDTB dataset
          if word not in self.vocab:
            # bottleneck
            continue

          value = np.asarray(line[-self.word_vector_width:], dtype=np.float32)

          vocab.append(word)
          embedding.append(value)

        # unk token taken as average of all other vectors
        vocab.insert(0, const.UNK)
        embedding.insert(0, np.mean(embedding, axis=0))

        # all 0 for padding
        vocab.insert(0, const.PAD)
        embedding.insert(0, np.zeros(self.word_vector_width, dtype=np.float))

        self.vocab = vocab

      else:

        import gensim

        googlenews_gz = os.path.join(
          const.W2V, "GoogleNews-vectors-negative300.bin.gz")

        tf.logging.info("Loading GoogleNews w2v. This will take some time.")
        model = gensim.models.KeyedVectors.load_word2vec_format(
          googlenews_gz, binary=True)
        tf.logging.info("done.")

        # self.vocab: sorted list of vocab
        vocab = []
        embedding = []
        for v in self.vocab:
          if v in model:
            embedding.append(model.get_vector(v))
            vocab.append(v)

        # unk token taken as average of all other vectors
        vocab.insert(0, const.UNK)
        embedding.insert(0, np.mean(embedding, axis=0))

        # all 0 for padding
        vocab.insert(0, const.PAD)
        embedding.insert(0, np.zeros(self.word_vector_width, dtype=np.float))

      self.vocab = vocab

    self._embedding_table = np.array(embedding, np.float)

  def get_embedding_table(self):
    return self.embedding_table

  def get_vocab(self):
    return self.vocab

  def convert_to_ids(self, examples, l2i):
    """

    Args:
      examples: List of `PDTBRelation` instances

    Returns:

    """
    arg1, arg2, conn, label_ids = [], [], [], []

    def convert_single_example(example):
      data = []
      for tokens in [example.arg1, example.arg2, example.conn]:
        token_ids = []
        if tokens:
          for token in tokens:
            token_ids.append(
              self.vocab.index(token) if token in self.vocab else 1) # UNK at 1
        else:
          # mainly targets connectives
          token_ids = [0] * self.max_arg_length # PAD at 0
        data.append(token_ids)

      label_id = l2i(example.label)
      data.append(label_id)

      return data

    for example in examples:
      feature = convert_single_example(example)
      arg1.append(feature[0])
      arg2.append(feature[1])
      conn.append(feature[2])
      label_ids.append(feature[3])

    return arg1, arg2, conn, label_ids

  def run(self, examples):
    raise NotImplementedError()

  def convert_to_values(self, examples):
    raise NotImplementedError()
