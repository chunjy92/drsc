#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .Model import Model

__author__ = 'Jayeol Chun'


class MLP(Model):
  def __init__(self,
               labels=None,
               max_arg_length=128,
               word_vector_width=50,
               hidden_size=300,
               num_hidden_layers=2,
               learning_rate=0.0001,
               optimizer='adam',
               sense_type='implicit',
               pooling_action='sum',
               conn_action=None,
               embedding=None,
               embedding_shape=None,
               do_pooling_first=False,
               finetune_embedding=False,
               scope=None):
    # model architecture will slightly vary depending on combinations of:
    # [dataset_type, pooling_action, conn_action]
    self.sense_type = sense_type # see const.DATASET_TYPES
    self.pooling_action = pooling_action # see const.POOLING_ACTIONS
    # pooling_action_split = pooling_action.split("_")
    # self.pooling_action = pooling_action_split[0] # sum, mean, concat, ..
    # self.pooling_timing = pooling_action_split[1] # first or later
    self.pooling_action = pooling_action
    self.do_pooling_first = do_pooling_first

    self.conn_action = conn_action # see const.CONN_ACTIONS

    # experimental settings
    self.max_arg_length = max_arg_length
    self.word_vector_width = word_vector_width
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.learning_rate = learning_rate
    self.optimizer = optimizer

    # data-related
    self.labels = labels
    self.num_labels = len(self.labels)

    # embedding related
    self.embedding = embedding
    self.embedding_shape = embedding_shape
    self.finetune_embedding = finetune_embedding

    # model architecture will slightly vary depending on `self.embedding`.
    self.build(scope)

  def build_dense_layers_single_input(self, input_tensor, num_layers=None):
    num_layers = num_layers if num_layers else self.num_hidden_layers

    output = input_tensor

    for i in range(num_layers):
      with tf.variable_scope(f"layer_{i}"):
        output = tf.layers.dense(
          name="dense",
          inputs=output,
          units=self.hidden_size,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
          activation=tf.nn.tanh
        )

    return output

  def combine_pooled_tensors(self, input_1, input_2, target_hidden_size=None,
                               add_bias=False):
    hidden_size = input_1.shape[-1].value
    target_hidden_size = \
      target_hidden_size if target_hidden_size else self.hidden_size

    weight_1 = tf.get_variable(
      f"combine_weight_{1}", [target_hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02),
      regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)
    )

    weight_2 = tf.get_variable(
      f"combine_weight_{2}", [target_hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02),
      regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)
    )

    input_1_matmul = tf.matmul(input_1, weight_1, transpose_b=True)
    input_2_matmul = tf.matmul(input_2, weight_2, transpose_b=True)
    logits = tf.add(input_1_matmul, input_2_matmul)

    if add_bias:
      bias = tf.get_variable(
        f"combine_bias", [target_hidden_size],
        initializer=tf.zeros_initializer()
      )
      logits = tf.nn.bias_add(logits, bias)

    return tf.nn.tanh(logits, name="combine_tanh")

  def build(self, scope=None):
    with tf.variable_scope(scope, default_name="mlp_model"):
      arg1, arg2 = self.build_input_pipeline()

      if self.sense_type == "implicit":
        if self.do_pooling_first:
          with tf.variable_scope("pooling"):
            arg1_pooled = \
              self.apply_pooling_fn(arg1, pooling_action=self.pooling_action)
            arg2_pooled = \
              self.apply_pooling_fn(arg2, pooling_action=self.pooling_action)

          combined = self.combine_pooled_tensors(arg1_pooled, arg2_pooled,
                                                 add_bias=True)

          output = self.build_dense_layers_single_input(combined)

        else:
          raise NotImplementedError(
            "Currently `do_pooling_first` must be specified")
      else:
        # TODO: for other sense types
        raise NotImplementedError(
          "Currently only `implicit` types are supported")


    hidden_size = output.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [self.num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02),
      regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    output_bias = tf.get_variable(
      "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      logits = tf.matmul(output, output_weights,
                         transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      self.preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

      one_hot_labels = tf.one_hot(self.label, depth=self.num_labels,
                                  dtype=tf.float32)

      self.correct = tf.cast(tf.equal(self.preds, self.label), "float")
      self.acc = tf.reduce_mean(self.correct, name="accuracy")

      self.per_example_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=one_hot_labels, logits=logits)
      loss = tf.reduce_mean(self.per_example_loss)

      # l2-norm
      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      reg_constant = 0.01
      self.loss = loss + reg_constant * tf.reduce_sum(reg_losses)

    optimizer = self.get_optimizer(self.optimizer)
    self.train_op = optimizer(self.learning_rate).minimize(self.loss)


  def postprocess_batch_ids(self, batch):
    arg1, arg2, conn, label_ids = batch

    feed_dict = {
      self.arg1: arg1,
      self.arg2: arg2,
      self.conn: conn,
      self.label: label_ids
    }

    return feed_dict

  def postprocess_batch_vals(self, batch, values,
                             l2i_mapping=None,
                             exid_to_feature_mapping=None):
    label_ids = []
    batch_bert_outputs = []
    for batch_example in batch:
      # exid indexes into values to fetch correct values
      batch_exid = batch_example.exid

      batch_bert_outputs.append(values[batch_exid])

      # label
      label_ids.append(l2i_mapping(batch_example.label))

    # prepare bert output: [batch, total_seq_length, bert_hidden_size]
    batch_bert_outputs = np.asarray(batch_bert_outputs)
    total_seq_length = batch_bert_outputs.shape[1]
    assert total_seq_length == self.max_arg_length*2, \
      "Sequence length mismatch between BERT output and parameter"

    arg1 = batch_bert_outputs[:, :self.max_arg_length, :]
    arg2 = batch_bert_outputs[:, self.max_arg_length:, :]

    # TODO: connectives
    # since we don't use connectives, set them 0 for now
    conn = np.zeros([len(batch),
                     self.max_arg_length,
                     batch_bert_outputs.shape[-1]])

    feed_dict = {
      self.arg1          : arg1,
      self.arg2          : arg2,
      self.conn          : conn,
      self.label         : label_ids,
    }

    return feed_dict
