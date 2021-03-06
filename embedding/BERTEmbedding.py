#! /usr/bin/python3
# -*- coding: utf-8 -*-
import copy

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
               model,
               bert_config_file,
               vocab_file,
               init_checkpoint,
               batch_size=32,
               max_seq_length=128,
               do_lower_case=False,
               finetune_embedding=False,
               split_args=False,
               is_training=False,
               truncation_mode="normal",
               padding_action='normal',
               scope=None):
    self.model = model
    self.is_mask_attentional_model = self.model.startswith("mask")
    self.bert_config_file = bert_config_file
    self.vocab_file = vocab_file
    self.init_checkpoint = init_checkpoint
    self.batch_size = batch_size
    self.max_seq_length = max_seq_length
    self.max_arg_length = int(max_seq_length/2)
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
    self.bert_config = copy.deepcopy(
      modeling.BertConfig.from_json_file(self.bert_config_file))

    self.is_training = is_training
    if not self.is_training:
      self.bert_config.hidden_dropout_prob = 0.0
      self.bert_config.attention_probs_dropout_prob = 0.0

    self._embedding_table = None
    self._vocab = tokenization.load_vocab(self.vocab_file)

    # max_position_embeddings==512
    if self.max_seq_length > self.bert_config.max_position_embeddings:
      raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (self.max_seq_length, self.bert_config.max_position_embeddings))

    self.build()

  ################################### BUILD ####################################
  def build_bert_model(self,
                       input_ids,
                       input_mask,
                       token_type_ids):
    with tf.variable_scope('bert'):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (embedding_output, _) = modeling.embedding_lookup(
            input_ids=input_ids,
            vocab_size=self.bert_config.vocab_size,
            embedding_size=self.bert_config.hidden_size,
            initializer_range=self.bert_config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=False
        )

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        embedding_output = modeling.embedding_postprocessor(
            input_tensor=embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=self.bert_config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=self.bert_config.initializer_range,
            max_position_embeddings=self.bert_config.max_position_embeddings,
            dropout_prob=self.bert_config.hidden_dropout_prob
        )

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = modeling.create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer, only fetching the final lyaer
        # `final_layer` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = modeling.transformer_model(
            input_tensor=embedding_output,
            attention_mask=attention_mask,
            hidden_size=self.bert_config.hidden_size,
            num_hidden_layers=self.bert_config.num_hidden_layers,
            num_attention_heads=self.bert_config.num_attention_heads,
            intermediate_size=self.bert_config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(
              self.bert_config.hidden_act),
            hidden_dropout_prob=self.bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=\
              self.bert_config.attention_probs_dropout_prob,
            initializer_range=self.bert_config.initializer_range,
            do_return_all_layers=True
        )

      self.sequence_output = self.all_encoder_layers[-1]

  def build_mask_attn(self, scope=None):
    tf.logging.debug("Building mask attention pipeline in BERT Embedding")
    self.arg = tf.placeholder(tf.int32, [None, self.max_seq_length],
                              name='arg')
    self.label = tf.placeholder(tf.int32, [None], name="label")

    # TODO: max_len for conn???
    self.conn = tf.placeholder(tf.int32, [None, self.max_seq_length],
                               name="conn")

    # for bert embedding
    self.arg_attn_mask = tf.placeholder(
      tf.int32, [None, self.max_seq_length], name="arg_attn_mask")

    self.segment_ids = tf.placeholder(
      tf.int32, [None, self.max_seq_length], name="segment_ids")

    bert_model = modeling.BertModel(config=self.bert_config,
                                    is_training=self.is_training,
                                    input_ids=self.arg,
                                    input_mask=self.arg_attn_mask,
                                    token_type_ids=self.segment_ids,
                                    use_one_hot_embeddings=False,
                                    scope='bert')
    self.bert_arg = bert_model.get_sequence_output()
    self.bert_mask_concat = self.arg_attn_mask

  def build_block_attn_single_arg(self, scope=None):
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

    # # custom
    # self.build_bert_model(input_ids=self.arg,
    #                       input_mask=self.arg_attn_mask,
    #                       token_type_ids=segment_ids)
    #
    # bert_arg = self.sequence_output

    input_shape = modeling.get_shape_list(bert_arg, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    self.bert_arg = tf.reshape(bert_arg,
                               shape=[batch_size/2, seq_length*2, width])
    self.bert_mask_concat = tf.reshape(self.arg_attn_mask,
                                       shape=[batch_size/2, seq_length*2])

  def build_block_attn_two_args(self, scope=None):
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
    self.bert_arg = bert_model.get_sequence_output()

    # # custom
    # self.build_bert_model(input_ids=arg_concat,
    #                       input_mask=self.bert_mask_concat,
    #                       token_type_ids=segment_ids)
    #
    # self.bert_arg_concat = self.sequence_output

  def build(self, scope=None):
    """"""
    if self.is_mask_attentional_model:
      self.build_mask_attn(scope)
    else:
      if self.split_args:
        self.build_block_attn_single_arg(scope)
      else:
        self.build_block_attn_two_args(scope)

  ############################### POSTPROCESS ##################################
  def convert_to_ids(self, examples, l2i, **kwargs):
    if self.is_mask_attentional_model:
      return self.convert_to_ids_mask_attn(examples, l2i, **kwargs)
    else:
      return self.convert_to_ids_block_attn(examples, l2i, **kwargs)

  def convert_to_ids_mask_attn(self, examples, l2i, **kwargs):
    args, conn, labels, arg_mask, arg_segment_ids = [], [], [], [], []

    for example in examples:
      tokens_a = self.tokenizer.tokenize(example.arg1)
      tokens_b = self.tokenizer.tokenize(example.arg2)

      _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length-3)

      tokens = []
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)

      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < self.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == self.max_seq_length
      assert len(input_mask) == self.max_seq_length
      assert len(segment_ids) == self.max_seq_length

      label_id = l2i(example.label)

      args.append(input_ids)
      # TODO: Connectives jsut padding values
      conn.append([self.vocab[const.PAD]] * self.max_seq_length)
      labels.append(label_id)
      arg_mask.append(input_mask)
      arg_segment_ids.append(segment_ids)

    return args, conn, labels, arg_mask, arg_segment_ids

  def convert_to_ids_block_attn(self, examples, l2i, **kwargs):
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

      # if not self.split_args:
      # retain CLS
      if len(tokens_b) > keep_length :
        if self.truncation_mode == 'normal':
          tokens_b = tokens_b[:keep_length]
        else:
          tokens_b = tokens_b[-keep_length:]
      tokens_b.insert(0, "[CLS]")
      tokens_b.append("[SEP]")
      # else:
      #   # NO CLS
      #   keep_length = self.max_arg_length - 1
      #   if len(tokens_b) > keep_length:
      #     if self.truncation_mode == 'normal':
      #       tokens_b = tokens_b[:keep_length]
      #     else:
      #       tokens_b = tokens_b[-keep_length:]
      #   tokens_b.append("[SEP]")

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
  def get_bert_arg(self):
    return self.bert_arg

  def get_attn_mask(self):
    return self.bert_mask_concat

  ############################### PLACEHOLDERS #################################
  def get_arg_placeholder(self):
    """only if self.split_arg == True"""
    return self.arg

  def get_arg_mask_placeholder(self):
    """only if self.split_arg == True"""
    return self.arg_attn_mask

  def get_arg1_placeholder(self):
    return self.arg1

  def get_arg2_placeholder(self):
    return self.arg2

  def get_label_placeholder(self):
    return self.label

  def get_conn_placeholder(self):
    return self.conn

  def get_arg1_mask_placeholder(self):
    return self.arg1_attn_mask

  def get_arg2_mask_placeholder(self):
    return self.arg2_attn_mask

  def get_segment_ids_placeholder(self):
    return self.segment_ids

  def get_all_placeholder_ops(self):
    ret = []
    if self.is_mask_attentional_model:
      ret = [self.arg, self.conn, self.label, self.arg_attn_mask,
             self.segment_ids]
    else:
      if self.split_args:
        ret = [self.arg, self.conn, self.label, self.arg_attn_mask]
      else:
        ret = [self.arg1, self.arg2, self.conn, self.label,
               self.arg1_attn_mask, self.arg2_attn_mask]

    return ret

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()