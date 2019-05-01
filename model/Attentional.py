#! /usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from bert import modeling

__author__ = 'Jayeol Chun'


class Attentional(object):
  def __init__(self,
               labels,
               max_arg_length=128,
               word_vector_width=768,
               hidden_size=768,
               num_hidden_layers=3,
               num_attention_heads=8,
               learning_rate=3e-4,
               optimizer='adam',
               attention_type="self_attn",
               sense_type='implicit',
               pooling_action='sum',
               conn_action=None,
               do_pooling_first=False,
               scope=None):
    # model architecture will slightly vary depending on combinations of:
    # [dataset_type, pooling_action, conn_action]
    self.attention_type = attention_type
    self.sense_type = sense_type  # see const.DATASET_TYPES
    self.pooling_action = pooling_action  # see const.POOLING_ACTIONS

    self.pooling_action = pooling_action
    self.do_pooling_first = do_pooling_first

    self.conn_action = conn_action  # see const.CONN_ACTIONS

    # experimental settings
    self.max_arg_length = max_arg_length
    self.word_vector_width = word_vector_width
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.learning_rate = learning_rate
    self.optimizer = optimizer

    # data-related
    self.labels = labels
    self.num_labels = len(self.labels)

    self.build(scope)

  def build_attn_layers_multiple_types(self, input_tensor, second_tensor):
    pass

  def build_attn_layers(self, input_tensor, second_tensor,
                        attention_mask=None,
                        intermediate_size=3072,
                        intermediate_act_fn=modeling.gelu,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1,
                        initializer_range=0.02,
                        do_return_all_layers=False):
    """See `attention_layer` defined in `bert/modeling.py`"""
    # input tensor shape: [batch, arg_length, BERT_hidden_size]
    # effectively: [32, 128, 768]
    attention_head_size = int(self.hidden_size / self.num_attention_heads)
    input_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
    tf.logging.info(f"Input Shape: {input_shape} {input_tensor.shape}")
    batch_size = input_shape[0]

    if self.attention_type == "self_attn":
      # [batch, arg_length * 2, BERT_hidden_size]
      input_tensor = tf.concat([input_tensor, second_tensor], axis=1)
      # from_tensor = to_tensor = modeling.reshape_to_matrix(input_tensor)
      # from_tensor_length = to_tensor_length = self.max_arg_length * 2
      prev_output = modeling.reshape_to_matrix(input_tensor)
      seq_length =self.max_arg_length*2

    elif self.attention_type == "inter_attn":
      raise NotImplementedError()
    elif self.attention_type in ["inter_intra_attn", "inter_self_attn"]:
      return self.build_attn_layers_multiple_types(input_tensor, second_tensor)

    # seq_length = input_shape[1] # self.max_arg_length
    # input_width = input_shape[2] # self.word_vector_width

    all_layer_outputs = []
    for i in range(self.num_hidden_layers):
      with tf.variable_scope(f"layer_{i}"):
        layer_input = prev_output

        with tf.variable_scope("attention"):
          attention_heads = []
          with tf.variable_scope(self.attention_type):
            attention_head = modeling.attention_layer(
              from_tensor=prev_output,
              to_tensor=prev_output,
              attention_mask=attention_mask,
              num_attention_heads=self.num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
            attention_heads.append(attention_head)

          attention_output = None
          if len(attention_heads) == 1:
            attention_output = attention_heads[0]
          else:
            # In the case where we have other sequences, we just concatenate
            # them to the self-attention head before the projection.
            attention_output = tf.concat(attention_heads, axis=-1)

          # Run a linear projection of `hidden_size` then add a residual
          # with `layer_input`.
          with tf.variable_scope("output"):
            attention_output = tf.layers.dense(
              attention_output,
              self.hidden_size,
              kernel_initializer=modeling.create_initializer(initializer_range))
            attention_output = modeling.dropout(attention_output,
                                                hidden_dropout_prob)
            attention_output = modeling.layer_norm(attention_output + layer_input)

        # The activation is only applied to the "intermediate" hidden layer.
        with tf.variable_scope("intermediate"):
          intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=modeling.create_initializer(initializer_range))

        # Down-project back to `hidden_size` then add the residual.
        with tf.variable_scope("output"):
          layer_output = tf.layers.dense(
            intermediate_output,
            self.hidden_size,
            kernel_initializer=modeling.create_initializer(initializer_range))
          layer_output = modeling.dropout(layer_output, hidden_dropout_prob)
          layer_output = modeling.layer_norm(layer_output + attention_output)
          prev_output = layer_output
          all_layer_outputs.append(layer_output)

    if do_return_all_layers:
      final_outputs = []
      for layer_output in all_layer_outputs:
        final_output = modeling.reshape_from_matrix(layer_output, input_shape)
        final_outputs.append(final_output)
      return final_outputs
    else:
      final_output = modeling.reshape_from_matrix(prev_output, input_shape)
      return final_output

  # def build_attn_layers_double_types

  def encode_concat_context(self):
    pass

  def build(self, scope=None):
    with tf.variable_scope(scope, default_name="attentional_model"):
      self.arg1 = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="arg1")
      self.arg2 = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="arg2")
      self.label = tf.placeholder(
        tf.int32, [None, self.num_labels], name="label")
      # TODO: max_len for conn???
      self.conn = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="conn")

      # placehodlers for attention_mask
      self.arg1_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg1_attention_mask")
      self.arg2_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg2_attention_mask")

      # segment ids (does not necessarily need to be a placeholder)

      # pos encoding
      # self.encode_concat_context()

      # attention mask # TODO
      # attention_mask = modeling.create_attention_mask_from_input_mask(
      #   input_ids, input_mask)

      # attention layers
      self.all_encoder_layers = \
        self.build_attn_layers(self.arg1, self.arg2)

      self.sequence_output = self.all_encoder_layers[-1]

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers
