#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

from .Model import Model

__author__ = 'Jayeol Chun'


class FeedForward(Model):
  ################################### BUILD ####################################
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
    )

    weight_2 = tf.get_variable(
      f"combine_weight_{2}", [target_hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02),
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
    with tf.variable_scope(scope, default_name="feedforward_model"):
      self.build_input_pipeline()

      if self.is_bert_embedding:
        arg_concat = self.embedding.get_bert_arg()
        combined_output = self.apply_cls_pooling_fn(arg_concat)
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

    logits = self.build_loss_op(output)

    self.preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32, name="preds")
    self.correct = tf.cast(tf.equal(self.preds, self.label), "float",
                           name="correct")
    self.acc = tf.reduce_mean(self.correct, name="acc")

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('acc', self.acc)

    self.train_op = self.build_train_op()

  ################################# POSTPROCESS ################################
  def postprocess_batch(self, batch, fetch_ops):
    arg1, arg2, conn, label_ids, arg1_mask, arg2_mask = batch

    feed_dict = None
    if self.is_bert_embedding:
      feed_dict = {
        self.arg1          : arg1,
        self.arg2          : arg2,
        self.conn          : conn,
        self.label         : label_ids,
        self.arg1_attn_mask: arg1_mask,
        self.arg2_attn_mask: arg2_mask
      }
    else:
      feed_dict = {
        self.arg1 : arg1,
        self.arg2 : arg2,
        self.conn : conn,
        self.label: label_ids
      }

    ops = self.fetch_ops(fetch_op_names=fetch_ops)

    return ops, feed_dict
