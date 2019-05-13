#! /usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf

from bert import modeling, tokenization
from utils import const
from .Embedding import Embedding

__author__ = 'Jayeol Chun'


class BERTEmbedding(Embedding):
  """Todo (May 11):
      2) split_arg_in_bert: whether to treat each arg as separate example
    """
  def __init__(self,
               bert_config_file,
               vocab_file,
               init_checkpoint,
               batch_size=32,
               max_arg_length=128,
               do_lower_case=False,
               finetune_embedding=False,
               split_args=False,
               is_training=False,
               truncation_mode="normal",
               padding_action='normal',
               scope=None):
    self.bert_config_file = bert_config_file
    self.vocab_file = vocab_file
    self.init_checkpoint = init_checkpoint
    self.batch_size = batch_size
    self.max_arg_length = max_arg_length
    self.do_lower_case = do_lower_case
    self.split_args = split_args
    self.finetune_embedding = finetune_embedding
    self.truncation_mode = truncation_mode
    self.padding_action = padding_action

    # Word-Piece tokenizer
    self.tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

    # load bert
    tokenization.validate_case_matches_checkpoint(self.do_lower_case,
                                                  self.init_checkpoint)
    self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)

    self.is_training = is_training

    self._embedding_table = None
    self._vocab = tokenization.load_vocab(self.vocab_file)

    # max_position_embeddings==512
    if self.max_arg_length > self.bert_config.max_position_embeddings:
      raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (self.max_arg_length, self.bert_config.max_position_embeddings))

    self.build()

  ################################### BUILD ####################################
  def build_single_arg(self, scope=None):
    tf.logging.debug("Building single arg pipeline in BERT Embedding")
    self.arg = tf.placeholder(tf.int32, [None, self.max_arg_length],
                              name='arg')
    self.label = tf.placeholder(tf.int32, [None], name="label")

    # TODO: max_len for conn???
    self.conn = tf.placeholder(tf.int32, [None, self.max_arg_length],
                               name="conn")

    self.arg_attn_mask = tf.placeholder(
      tf.int32, [None, self.max_arg_length], name="arg_attn_mask")

    segment_ids = tf.zeros_like(self.arg, dtype=tf.int32)

    bert_model = modeling.BertModel(config=self.bert_config,
                                    is_training=self.is_training,
                                    input_ids=self.arg,
                                    input_mask=self.arg_attn_mask,
                                    token_type_ids=segment_ids,
                                    use_one_hot_embeddings=False,
                                    scope='bert')
    bert_arg = bert_model.get_sequence_output()

    input_shape = modeling.get_shape_list(bert_arg, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    self.bert_arg_concat = tf.reshape(bert_arg,
                                      shape=[batch_size/2, seq_length*2, width])
    self.bert_mask_concat = tf.reshape(self.arg_attn_mask,
                                       shape=[batch_size/2, seq_length*2])

  def build_two_args(self, scope=None):
    tf.logging.debug("Building two args pipeline in BERT Embedding")
    self.arg1 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                               name="arg1")
    self.arg2 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                               name="arg2")
    self.label = tf.placeholder(tf.int32, [None], name="label")

    # TODO: max_len for conn???
    self.conn = tf.placeholder(tf.int32, [None, self.max_arg_length],
                               name="conn")

    self.arg1_attn_mask = tf.placeholder(
      tf.int32, [None, self.max_arg_length], name="arg1_attn_mask")
    self.arg2_attn_mask = tf.placeholder(
      tf.int32, [None, self.max_arg_length], name="arg2_attn_mask")

    # inputs to bert model
    arg_concat = tf.concat([self.arg1, self.arg2], axis=1)
    self.bert_mask_concat = \
      tf.concat([self.arg1_attn_mask, self.arg2_attn_mask], axis=1)

    segment_ids = tf.concat([
      tf.zeros_like(self.arg1, dtype=tf.int32),  # Arg1: 0s
      tf.ones_like(self.arg2, dtype=tf.int32)  # Arg2: 1s
    ], axis=1)

    bert_model = modeling.BertModel(config=self.bert_config,
                                    is_training=self.is_training,
                                    input_ids=arg_concat,
                                    input_mask=self.bert_mask_concat,
                                    token_type_ids=segment_ids,
                                    use_one_hot_embeddings=False,
                                    scope='bert')

    # [batch, arg_len*2, hidden_size]
    self.bert_arg_concat = bert_model.get_sequence_output()

  def build(self, scope=None):
    """"""
    if self.split_args:
      self.build_single_arg(scope)
    else:
      self.build_two_args(scope)

  ############################### POSTPROCESS ##################################
  def convert_to_ids(self, examples, l2i, **kwargs):
    arg1, arg2, conn, labels, arg1_mask, arg2_mask = [], [], [], [], [], []

    for example in examples:
      tokens_a = self.tokenizer.tokenize(example.arg1)
      tokens_b = self.tokenizer.tokenize(example.arg2)

      keep_length = self.max_arg_length - 2
      if len(tokens_a) > keep_length:
        if self.truncation_mode == 'normal':
          tokens_a = tokens_a[:keep_length]
        else:
          tokens_a = tokens_a[-keep_length:]
      tokens_a.insert(0, "[CLS]")
      tokens_a.append("[SEP]")

      if not self.split_args:
        # retain CLS
        if len(tokens_b) > keep_length :
          if self.truncation_mode == 'normal':
            tokens_b = tokens_b[:keep_length]
          else:
            tokens_b = tokens_b[-keep_length:]
        tokens_b.insert(0, "[CLS]")
        tokens_b.append("[SEP]")
      else:
        # NO CLS
        keep_length = self.max_arg_length - 1
        if len(tokens_b) > keep_length:
          if self.truncation_mode == 'normal':
            tokens_b = tokens_b[:keep_length]
          else:
            tokens_b = tokens_b[-keep_length:]
        tokens_b.append("[SEP]")

      input_ids_a = self.tokenizer.convert_tokens_to_ids(tokens_a)
      input_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask_a = [1] * len(input_ids_a)
      while len(input_ids_a) < self.max_arg_length:
        if self.padding_action == 'pad_left_arg1':
          input_ids_a.insert(1, 0)
          input_mask_a.insert(1, 0)
        else:
          input_ids_a.append(0)
          input_mask_a.append(0)

      input_mask_b = [1] * len(input_ids_b)
      while len(input_ids_b) < self.max_arg_length:
        input_ids_b.append(0)
        input_mask_b.append(0)

      label_id = l2i(example.label)

      # into data
      arg1.append(input_ids_a)
      arg2.append(input_ids_b)
      labels.append(label_id)
      # TODO: Connectives jsut padding values
      conn.append([self.vocab[const.PAD]] * self.max_arg_length)
      arg1_mask.append(input_mask_a)
      arg2_mask.append(input_mask_b)

    return arg1, arg2, conn, labels, arg1_mask, arg2_mask

  ################################## GETTERS ###################################
  def get_arg_concat(self):
    return self.bert_arg_concat

  def get_attn_mask(self):
    return self.bert_mask_concat

  ############################### PLACEHOLDERS #################################
  def get_arg(self):
    """only if self.split_arg == True"""
    return self.arg

  def get_arg_mask(self):
    """only if self.split_arg == True"""
    return self.arg_attn_mask

  def get_arg1(self):
    return self.arg1

  def get_arg2(self):
    return self.arg2

  def get_label(self):
    return self.label

  def get_conn(self):
    return self.conn

  def get_arg1_mask(self):
    return self.arg1_attn_mask

  def get_arg2_mask(self):
    return self.arg2_attn_mask

  def get_all_placeholder_ops(self):
    ret = []
    if self.split_args:
      ret = [self.arg, self.conn, self.label, self.arg_attn_mask]
    else:
      ret = [self.arg1, self.arg2, self.conn, self.label,
             self.arg1_attn_mask, self.arg2_attn_mask]

    return ret
