#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .Model import Model

__author__ = 'Jayeol Chun'


class MLP(Model):
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
      self.build_input_pipeline()

      if self.is_finetunable_bert_embedding:
        # finetunable bert embedding concatenates arg1 and arg2
        arg_concat = self.embedding.get_arg_concat()
        combined_output = self.apply_cls_pooling_fn(arg_concat)
      else:
        if self.is_bert_embedding:
          arg1, arg2 = self.arg1, self.arg2
        else:
          self.embedding_table = self.init_embedding(self.embedding_placeholder)

          # embedding lookup
          with tf.variable_scope("embedding"):
            arg1 = tf.nn.embedding_lookup(self.embedding_table, self.arg1)
            arg2 = tf.nn.embedding_lookup(self.embedding_table, self.arg2)

        with tf.variable_scope("pooling"):
          arg1_pooled = self.apply_pooling_fn(arg1)
          arg2_pooled = self.apply_pooling_fn(arg2)

        # linear combination for static embeddings
        combined_output = self.combine_pooled_tensors(arg1_pooled, arg2_pooled,
                                                      add_bias=True)

      # same dense layers
      output = self.build_dense_layers_single_input(combined_output)

    hidden_size = output.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [self.num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02),
      regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

    output_bias = tf.get_variable(
      "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      logits = tf.matmul(output, output_weights, transpose_b=True)
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

      tf.summary.scalar('loss', self.loss)
      tf.summary.scalar('acc', self.acc)

    self.train_op = self.optimizer(self.learning_rate).minimize(self.loss)

  ################################# POSTPROCESS ################################
  def postprocess_batch_ids(self, batch):
    arg1, arg2, conn, label_ids, arg1_mask, arg2_mask = batch

    feed_dict = None
    if self.is_finetunable_bert_embedding:
      arg1_attn_mask = arg1_mask
      arg2_attn_mask = arg2_mask

      if not arg1_attn_mask:
        arg1_attn_mask = []

        for arg1_ids in arg1:
          arg1_mask = []
          for arg1_id in arg1_ids:
            if arg1_id == 0:  # PAD token id: 0
              arg1_mask.append(0)
            else:
              arg1_mask.append(1)
          arg1_attn_mask.append(arg1_mask)

      if not arg2_attn_mask:
        arg2_attn_mask = []

        for arg2_ids in arg2:
          arg2_mask = []
          for arg2_id in arg2_ids:
            if arg2_id == 0:  # PAD token id: 0
              arg2_mask.append(0)
            else:
              arg2_mask.append(1)
          arg2_attn_mask.append(arg2_mask)

      feed_dict = {
        self.arg1          : arg1,
        self.arg2          : arg2,
        self.conn          : conn,
        self.label         : label_ids,
        self.arg1_attn_mask: arg1_attn_mask,
        self.arg2_attn_mask: arg2_attn_mask
      }
    else:
      feed_dict = {
        self.arg1 : arg1,
        self.arg2 : arg2,
        self.conn : conn,
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
      self.arg1 : arg1,
      self.arg2 : arg2,
      self.conn : conn,
      self.label: label_ids,
    }

    return feed_dict
