#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

from bert import modeling
from .Model import Model

__author__ = 'Jayeol Chun'


class MaskAttentional(Model):
  ################################### BUILD ####################################
  def build_attn_layer(self,
                       input_tensor,
                       layer_attn_type,
                       attn_mask_concat,
                       segment_ids=None,
                       num_attention_heads=1,
                       size_per_head=512,
                       attention_probs_dropout_prob=0.1,
                       initializer_range=0.02,
                       do_return_2d_tensor=False):
    # TODO (May 5): To capture each softmax output, will need a modified
    #  `attention_layer`
    input_tensor_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_tensor_shape[0]
    total_seq_length = input_tensor_shape[1]

    attention_head = None
    if layer_attn_type == "self":
      attn_mask = modeling.create_attention_mask_from_input_mask(
        input_tensor, attn_mask_concat)

      attention_head = modeling.attention_layer(
        from_tensor=input_tensor,
        to_tensor=input_tensor,
        attention_mask=attn_mask,
        num_attention_heads=num_attention_heads,
        size_per_head=size_per_head,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        initializer_range=initializer_range,
        do_return_2d_tensor=do_return_2d_tensor,
        batch_size=batch_size,
        from_seq_length=total_seq_length,
        to_seq_length=total_seq_length
      )

    else:
      arg1_attn_mask = attn_mask_concat - segment_ids
      arg1_attn_mask = modeling.create_attention_mask_from_input_mask(
        input_tensor, arg1_attn_mask)

      arg2_attn_mask = segment_ids
      arg2_attn_mask = modeling.create_attention_mask_from_input_mask(
        input_tensor, arg2_attn_mask)

      with tf.variable_scope("to_arg1"):
        to_arg1 = modeling.attention_layer(
          from_tensor=input_tensor,
          to_tensor=input_tensor,
          attention_mask=arg1_attn_mask,
          num_attention_heads=num_attention_heads,
          size_per_head=size_per_head,
          attention_probs_dropout_prob=attention_probs_dropout_prob,
          initializer_range=initializer_range,
          do_return_2d_tensor=do_return_2d_tensor,
          batch_size=batch_size,
          from_seq_length=total_seq_length,
          to_seq_length=total_seq_length
        )


      with tf.variable_scope("to_arg2"):
        to_arg2 = modeling.attention_layer(
          from_tensor=input_tensor,
          to_tensor=input_tensor,
          attention_mask=arg2_attn_mask,
          num_attention_heads=num_attention_heads,
          size_per_head=size_per_head,
          attention_probs_dropout_prob=attention_probs_dropout_prob,
          initializer_range=initializer_range,
          do_return_2d_tensor=do_return_2d_tensor,
          batch_size=batch_size,
          from_seq_length=total_seq_length,
          to_seq_length=total_seq_length
        )


      attention_head = tf.add(to_arg1, to_arg2)

    return attention_head

  def build_attn_layers(self,
                        input_tensor,
                        attn_mask_concat,
                        segment_ids=None,
                        intermediate_size=2048,
                        intermediate_act_fn=modeling.gelu,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1,
                        initializer_range=0.02,
                        do_return_all_layers=False):
    """See `attention_layer` defined in `bert/modeling.py`"""
    if not self.is_training:
      hidden_dropout_prob = 0.0
      attention_probs_dropout_prob = 0.0

    # input tensor shape: [batch, arg_length, BERT_hidden_size]
    # for example, using default hparams vals: [64, 128, 768]
    attention_head_size = int(self.hidden_size / self.num_attention_heads)
    input_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
    prev_output = input_tensor

    all_layer_outputs = []
    for layer_idx in range(self.num_hidden_layers):
      with tf.variable_scope(f"layer_{layer_idx}"):
        layer_input = prev_output

        attn_type_split = self.attention_type.split("_")
        attn_type = " ".join(attn_type_split[:-1])
        tf.logging.info(f"{attn_type.capitalize()} Attention at "
                        f"{layer_idx}th Layer")

        attention_heads = []
        with tf.variable_scope(self.attention_type):
          attention_head = self.build_attn_layer(
            input_tensor=input_tensor,
            attn_mask_concat=attn_mask_concat,
            segment_ids=segment_ids,
            layer_attn_type=self.attention_type,
            num_attention_heads=self.num_attention_heads,
            size_per_head=attention_head_size,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=False
          )

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

  def build(self, scope=None):
    if not self.is_training:
      self.hidden_dropout_prob = 0.0
      self.attention_probs_dropout_prob = 0.0

    with tf.variable_scope(scope, default_name="attentional_model"):
      self.build_input_pipeline()

      if self.is_bert_embedding:
        input_concat = self.embedding.get_bert_arg()
        mask_concat = self.embedding.get_attn_mask()

      else:
        self.embedding_table = self.init_embedding(self.embedding_placeholder)

        # embedding lookup
        with tf.variable_scope("embedding"):
          arg1 = tf.nn.embedding_lookup(self.embedding_table, self.arg1)
          arg2 = tf.nn.embedding_lookup(self.embedding_table, self.arg2)

        input_concat = tf.concat([arg1, arg2], axis=1)
        mask_concat = tf.concat([self.arg1_attn_mask, self.arg2_attn_mask],
                                axis=1)

      # if word_vector_width and hidden_size do not match, need to project
      if self.word_vector_width != self.hidden_size:
        with tf.variable_scope("bert_projection"):
          input_concat = tf.layers.dense(
            name="dense",
            inputs=input_concat,
            units=self.hidden_size,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            use_bias=False
          )

      with tf.variable_scope("encoder"):
        # attention layers, for now keeping all encoder layers
        self.all_encoder_layers = \
          self.build_attn_layers(input_concat,
                                 attn_mask_concat=mask_concat,
                                 segment_ids=self.segment_ids,
                                 hidden_dropout_prob=self.hidden_dropout_prob,
                                 do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]

      with tf.variable_scope("pooler"):
        # see `CLS_ACTIONS` defined in `const.py`
        pooled_tensor = self.apply_cls_pooling_fn(self.sequence_output)

        pooled_output = tf.layers.dense(
          pooled_tensor,
          self.hidden_size,
          activation=tf.tanh,
          kernel_initializer=modeling.create_initializer()
        )

    logits = self.build_loss_op(pooled_output)

    self.preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32, name="preds")
    self.correct = tf.cast(tf.equal(self.preds, self.label), "float",
                           name="correct")
    self.acc = tf.reduce_mean(self.correct, name="acc")

    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.acc)

    self.train_op = self.build_train_op()

  ################################## GETTERS ###################################
  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  ################################# POSTPROCESS ################################
  def postprocess_batch(self, batch, fetch_ops):
    args, conn, label_ids, arg_mask, segment_ids = batch

    feed_dict = {
      self.arg: args,
      self.conn : conn,
      self.label: label_ids,
      self.arg_attn_mask: arg_mask,
      self.segment_ids: segment_ids
    }
    ops = self.fetch_ops(fetch_op_names=fetch_ops)

    return ops, feed_dict
