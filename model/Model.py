#! /usr/bin/python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import tensorflow as tf

__author__ = 'Jayeol Chun'


class Model(ABC):

  @abstractmethod
  def build(self):
    pass

  ################################# POSTPROCESS ################################
  @abstractmethod
  def postprocess_batch_ids(self, batch):
    pass

  @abstractmethod
  def postprocess_batch_vals(self, batch, values, **kwargs):
    pass

  ############################## PLACEHOLDER OPS ################################
  def init_embedding(self, placeholder):
    embedding_table = tf.get_variable(
      name="embedding_table",
      shape=self.embedding_shape,
      trainable=self.finetune_embedding
    )

    self.embedding_init_op = embedding_table.assign(placeholder)
    return embedding_table

  def build_input_pipeline(self):

    arg1, arg2 = None, None
    if self.embedding == 'bert':
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

      arg1 = self.arg1
      arg2 = self.arg2

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

      self.embedding_table = self.init_embedding(self.embedding_placeholder)

      # embedding lookup
      with tf.variable_scope("embedding"):
        arg1 = tf.nn.embedding_lookup(self.embedding_table, self.arg1)
        arg2 = tf.nn.embedding_lookup(self.embedding_table, self.arg2)

    return arg1, arg2

  ################################### UTIL #####################################
  def apply_pooling_fn(self, input_tensor, second_tensor=None,
                       pooling_action=None):
    # tensor shape: [batch, arg_length, word_vector_width]
    if pooling_action == "sum":
      if second_tensor:
        return tf.add(input_tensor, second_tensor)
      return tf.reduce_sum(input_tensor, axis=1)
    elif pooling_action == "mean":
      if second_tensor:
        return tf.reduce_mean([input_tensor, second_tensor], axis=0)
      return tf.reduce_mean(input_tensor, axis=1)
    elif pooling_action == "max":
      if second_tensor:
        return tf.reduce_max([input_tensor, second_tensor], axis=0)
      return tf.reduce_max(input_tensor, axis=1)
    elif pooling_action in ["concat", 'matmul']:
      # usually works on model outputs for each arg1 and arg2
      if not second_tensor:
        raise ValueError("Second tensor passed as `None` value")
      input_tensor_shape = input_tensor.shape
      second_tensor_shape = second_tensor.shape
      assert_op = tf.assert_equal(input_tensor_shape, second_tensor_shape)
      with tf.control_dependencies([assert_op]):
        if pooling_action == 'concat':
          return tf.concat([input_tensor, second_tensor], axis=-1)
        else:
          return tf.multiply(input_tensor, second_tensor)
    else:
      raise ValueError(f"{pooling_action} pooling function not understood")

  def get_optimizer(self, optimizer):
    if optimizer == "adam":
      return tf.train.AdamOptimizer
    elif optimizer == "sgd":
      return tf.train.GradientDescentOptimizer
    elif optimizer == 'adagrad': # Te's optimizer
      return tf.train.AdagradOptimizer
    else:
      raise ValueError(f"{optimizer} optimizer value not understood")
