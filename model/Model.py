#! /usr/bin/python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import tensorflow as tf

__author__ = 'Jayeol Chun'


class Model(ABC):
  def __init__(self,
               labels=None,
               attention_type=None,
               max_arg_length=128,
               word_vector_width=768,
               hidden_size=768,
               num_hidden_layers=3,
               num_attention_heads=8,
               hidden_dropout_prob=0.1,
               learning_rate=3e-4,
               embedding=None,
               embedding_name=None,
               embedding_shape=None,
               is_training=False,
               optimizer='adam',
               sense_type='implicit',
               pooling_action='concat',
               cls_action='first_cls',
               conn_action=None,
               do_pooling_first=False,
               finetune_embedding=False,
               scope=None):
    # model architecture will slightly vary depending on combinations of:
    # [dataset_type, pooling_action, conn_action]
    self.attention_type = attention_type
    self.sense_type = sense_type  # see const.DATASET_TYPES
    self.pooling_action = pooling_action  # see const.POOLING_ACTIONS
    self.embedding = embedding
    self.embedding_name = embedding_name
    self.embedding_shape = embedding_shape

    # model settings
    self.max_arg_length = max_arg_length
    self.word_vector_width = word_vector_width
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_dropout_prob = hidden_dropout_prob
    self.is_training = is_training

    # optimizer
    self.learning_rate = learning_rate
    self.optimizer_name = optimizer
    self.optimizer = self.get_optimizer(self.optimizer_name)

    # data-related
    self.labels = labels
    self.num_labels = len(self.labels)

    # control flags
    self.pooling_action = pooling_action
    self.do_pooling_first = do_pooling_first
    self.finetune_embedding = finetune_embedding
    self.cls_action = cls_action
    self.conn_action = conn_action  # see const.CONN_ACTIONS

    # boolean flags for embedding type
    self.is_bert_embedding = self.embedding_name == 'bert'
    self.is_finetunable_bert_embedding = \
      self.is_bert_embedding and self.finetune_embedding

    self.build(scope)

  ################################### BUILD ####################################
  @abstractmethod
  def build(self, scope=None):
    pass

  ############################### POSTPROCESS ##################################
  @abstractmethod
  def postprocess_batch_ids(self, batch):
    pass

  @abstractmethod
  def postprocess_batch_vals(self, batch, values, **kwargs):
    pass

  ############################## PLACEHOLDER OPS ###############################
  def init_embedding(self, placeholder):

    embedding_table = tf.get_variable(
      name="embedding_table",
      shape=self.embedding_shape,
      trainable=self.finetune_embedding
    )

    self.embedding_init_op = embedding_table.assign(placeholder)

    return embedding_table

  def build_input_pipeline(self):
    """Build all placeholder ops"""

    if self.is_finetunable_bert_embedding:
      placeholer_ops = self.embedding.get_all_placeholder_ops()
      self.arg1 = placeholer_ops[0]
      self.arg2 = placeholer_ops[1]
      self.conn = placeholer_ops[2]
      self.label = placeholer_ops[3]
      self.arg1_attn_mask = placeholer_ops[4]
      self.arg2_attn_mask = placeholer_ops[5]

      self.embedding_placeholder = None
      self.embedding_table = None

    elif self.is_bert_embedding:
      self.arg1 = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="arg1")
      self.arg2 = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="arg2")
      self.label = tf.placeholder(tf.int32, [None], name="label")

      # TODO: max_len for conn???
      self.conn = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="conn")

      # placehodlers for attention_mask
      self.arg1_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg1_attention_mask")
      self.arg2_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg2_attention_mask")

      self.embedding_placeholder = None
      self.embedding_table = None

    else:
      self.arg1 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="arg1")
      self.arg2 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="arg2")
      self.label = tf.placeholder(tf.int32, [None], name="label")

      # TODO: max_len for conn???
      self.conn = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="conn")

      self.embedding_placeholder = \
        tf.placeholder(tf.float32, self.embedding_shape,
                       "embedding_placeholder")

      # placehodlers for attention_mask
      self.arg1_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg1_attention_mask")
      self.arg2_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg2_attention_mask")

  ################################### UTIL #####################################
  def apply_cls_pooling_fn(self, input_tensor):
    first_cls = tf.squeeze(input_tensor[:, 0:1, :], axis=1)
    second_cls = tf.squeeze(
      input_tensor[:, self.max_arg_length:self.max_arg_length+1, :], axis=1)

    pooled_tensor = None
    if self.cls_action == "first_cls":
      pooled_tensor = first_cls
    elif self.cls_action == "second_cls":
      pooled_tensor = second_cls
    elif self.pooling_action == 'concat':
      pooled_tensor = tf.concat([first_cls, second_cls], axis=-1)
    elif self.pooling_action == 'new_cls':
      # TODO: requires re-structuring
      raise NotImplementedError("Currently `new_cls` is not supported")
    else:
      raise ValueError("CLS Pooling action not understood")

    return pooled_tensor

  def apply_pooling_fn(self, input_tensor, second_tensor=None):
    # tensor shape: [batch, arg_length, word_vector_width]
    if self.pooling_action == "sum":
      if second_tensor:
        return tf.add(input_tensor, second_tensor)
      return tf.reduce_sum(input_tensor, axis=1)
    elif self.pooling_action == "mean":
      if second_tensor:
        return tf.reduce_mean([input_tensor, second_tensor], axis=0)
      return tf.reduce_mean(input_tensor, axis=1)
    elif self.pooling_action == "max":
      if second_tensor:
        return tf.reduce_max([input_tensor, second_tensor], axis=0)
      return tf.reduce_max(input_tensor, axis=1)
    elif self.pooling_action in ["concat", 'matmul']:
      # usually works on model outputs for each arg1 and arg2
      if not second_tensor:
        raise ValueError("Second tensor passed as `None` value")
      input_tensor_shape = input_tensor.shape
      second_tensor_shape = second_tensor.shape
      assert_op = tf.assert_equal(input_tensor_shape, second_tensor_shape)
      with tf.control_dependencies([assert_op]):
        if self.pooling_action == 'concat':
          return tf.concat([input_tensor, second_tensor], axis=-1)
        else:
          return tf.multiply(input_tensor, second_tensor)
    else:
      raise ValueError("Pooling function not understood")

  def get_optimizer(self, optimizer):
    if optimizer == "adam":
      return tf.train.AdamOptimizer
    elif optimizer == "sgd":
      return tf.train.GradientDescentOptimizer
    elif optimizer == 'adagrad': # Te's optimizer
      return tf.train.AdagradOptimizer
    else:
      raise ValueError("Optimizer value not understood")
