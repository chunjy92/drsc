#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from bert import modeling, optimization
from .Model import Model

__author__ = 'Jayeol Chun'


class Attentional(Model):
  def build_attn_layer(self,
                       input_tensor,
                       attn_mask_concat,
                       layer_attn_type,
                       num_attention_heads=1,
                       size_per_head=512,
                       attention_probs_dropout_prob=0.1,
                       initializer_range=0.02,
                       do_return_2d_tensor=None):
    # TODO (May 5): To capture each softmax output, will need a modified
    #  `attention_layer`
    input_tensor_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_tensor_shape[0]
    total_seq_length = input_tensor_shape[1]
    arg_seq_length = int(total_seq_length / 2)

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
      arg1 = input_tensor[:, :arg_seq_length, :]
      arg2 = input_tensor[:, arg_seq_length:, :]

      arg1_attn_mask = attn_mask_concat[:, :arg_seq_length]
      arg1_attn_mask = modeling.create_attention_mask_from_input_mask(
        arg1, arg1_attn_mask)

      arg2_attn_mask = attn_mask_concat[:, arg_seq_length:]
      arg2_attn_mask = modeling.create_attention_mask_from_input_mask(
        arg2, arg2_attn_mask)

      if layer_attn_type == "inter":
        with tf.variable_scope("arg1_arg2"):
          arg1_to_arg2 = modeling.attention_layer(
            from_tensor=arg1,
            to_tensor=arg2,
            attention_mask=arg2_attn_mask,
            num_attention_heads=num_attention_heads,
            size_per_head=size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=do_return_2d_tensor,
            batch_size=batch_size,
            from_seq_length=arg_seq_length,
            to_seq_length=arg_seq_length
          )

        with tf.variable_scope("arg2_arg1"):
          arg2_to_arg1 = modeling.attention_layer(
            from_tensor=arg2,
            to_tensor=arg1,
            attention_mask=arg1_attn_mask,
            num_attention_heads=num_attention_heads,
            size_per_head=size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=do_return_2d_tensor,
            batch_size=batch_size,
            from_seq_length=arg_seq_length,
            to_seq_length=arg_seq_length
          )

        attention_head = tf.concat([arg1_to_arg2, arg2_to_arg1], axis=1)
      else:
        with tf.variable_scope("arg1_arg1"):
          arg1_to_arg1 = modeling.attention_layer(
            from_tensor=arg1,
            to_tensor=arg1,
            attention_mask=arg1_attn_mask,
            num_attention_heads=num_attention_heads,
            size_per_head=size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=do_return_2d_tensor,
            batch_size=batch_size,
            from_seq_length=arg_seq_length,
            to_seq_length=arg_seq_length
          )

        with tf.variable_scope("arg2_arg2"):
          arg2_to_arg2 = modeling.attention_layer(
            from_tensor=arg2,
            to_tensor=arg2,
            attention_mask=arg2_attn_mask,
            num_attention_heads=num_attention_heads,
            size_per_head=size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            initializer_range=initializer_range,
            do_return_2d_tensor=do_return_2d_tensor,
            batch_size=batch_size,
            from_seq_length=arg_seq_length,
            to_seq_length=arg_seq_length
          )

        attention_head = tf.concat([arg1_to_arg1, arg2_to_arg2], axis=1)

    return attention_head

  def build_attn_layers(self,
                        input_tensor,
                        attn_mask_concat,
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

    attention_type_split = self.attention_type.split("_")

    all_layer_outputs = []
    for layer_idx in range(self.num_hidden_layers):
      with tf.variable_scope(f"layer_{layer_idx}"):
        layer_input = prev_output

        if len(attention_type_split) == 3:
          indexer = layer_idx % 2
        else:  # len(attention_type_split) == 2:
          indexer = 0
        layer_attn_type = attention_type_split[indexer]

        tf.logging.info(
          f"{layer_attn_type.capitalize()} Attention at {layer_idx}th Layer")

        attention_heads = []
        with tf.variable_scope(f"{layer_attn_type}_attn"):
          attention_head = self.build_attn_layer(
            input_tensor=input_tensor,
            attn_mask_concat=attn_mask_concat,
            layer_attn_type=layer_attn_type,
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

  def encode_concat_context(self,
                            input_tensor,
                            segment_ids,
                            segment_vocab_size=16,
                            max_position_embeddings=512,
                            hidden_dropout_prob=0.1,
                            initializer_range=0.02,
                            use_segment_ids=False,
                            use_position_embedding=False):
    """See `embedding_postprocessor` defined in `bert/modeling.py`"""
    input_shape = modeling.get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_segment_ids:
      segment_table = tf.get_variable(
        name="segment_embeddings",
        shape=[segment_vocab_size, width],
        initializer=modeling.create_initializer(initializer_range))

      flat_segment_ids = tf.reshape(segment_ids, [-1]) # flatten
      one_hot_ids = tf.one_hot(flat_segment_ids, depth=segment_vocab_size)
      segment_embeddings = tf.matmul(one_hot_ids, segment_table)
      segment_embeddings = tf.reshape(segment_embeddings,
                                      [batch_size, seq_length, width])
      output += segment_embeddings

    if use_position_embedding:
      position_embeddings = tf.get_variable(
        name="position_embeddings",
        shape=[max_position_embeddings, width],
        initializer=modeling.create_initializer(initializer_range))

      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

    output = modeling.layer_norm_and_dropout(output, hidden_dropout_prob)
    return output

  def build(self, scope=None):
    if not self.is_training:
      self.hidden_dropout_prob = 0.0
      self.attention_probs_dropout_prob = 0.0

    with tf.variable_scope("attentional_model"):
      self.build_input_pipeline()

      if self.is_finetunable_bert_embedding:
        input_concat = self.embedding.get_arg_concat()
        mask_concat = self.embedding.get_attn_mask()

      else:
        if self.is_bert_embedding:
          arg1, arg2 = self.arg1, self.arg2
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

      # if not self.is_finetunable_bert_embedding:
      # additional context encoding with segment_ids and positional encoding
      # ONLY when BERT is not being fine-tuned
      batch_size = modeling.get_shape_list(input_concat, expected_rank=3)[0]
      segment_ids = tf.concat([
        tf.zeros([batch_size, self.max_arg_length], dtype=tf.int32),
        tf.ones([batch_size, self.max_arg_length], dtype=tf.int32)
      ], axis=1)

      input_concat = self.encode_concat_context(
        input_concat, segment_ids,
        hidden_dropout_prob=self.hidden_dropout_prob, use_segment_ids=True,
        use_position_embedding=True)

      # attention layers, for now keeping all encoder layers
      self.all_encoder_layers = \
        self.build_attn_layers(input_concat,
                               attn_mask_concat=mask_concat,
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
          kernel_initializer=modeling.create_initializer())

    # loss function
    hidden_size = pooled_output.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [self.num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      if self.is_training:
        # I.e., 0.1 dropout
        pooled_output = tf.nn.dropout(pooled_output, keep_prob=0.9)

      logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)

      # self.per_example_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      #   labels=one_hot_labels, logits=logits)
      self.per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.label, logits=logits, name="per_loss")
      self.loss = tf.reduce_mean(self.per_example_loss, name="mean_loss")

      # tf.summary.scalar('loss', self.loss)
      # tf.summary.scalar('acc', self.acc)

    self.preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32, name="preds")
    self.correct = tf.cast(tf.equal(self.preds, self.label), "float",
                           name="correct")
    self.acc = tf.reduce_mean(self.correct, name="acc")

    # self.train_op = \
    #   self.optimizer(self.learning_rate).minimize(self.loss, name="train_op")

    if self.optimizer_name == "adam" and self.num_warmup_steps > 0:
      tf.logging.info("Adam Weight Decay Optimizer")
      self.train_op = optimization.create_optimizer(
        loss=self.loss, init_lr=self.learning_rate,
        num_train_steps=self.num_train_steps,
        num_warmup_steps=self.num_warmup_steps,
        use_tpu=False
      )
    else:
      optimizer = self.get_optimizer(self.optimizer_name)
      tf.logging.info(f"{self.optimizer_name.capitalize()} Optimizer")
      self.train_op = optimizer(self.learning_rate).minimize(self.loss,
                                                             name='train_op')

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
  def postprocess_batch_ids(self, batch, fetch_ops):

    arg1, arg2, conn, label_ids, arg1_mask, arg2_mask = batch

    arg1_attn_mask = arg1_mask
    arg2_attn_mask = arg2_mask

    # TODO (May 11): do this in `PDTBProcessor` ################################
    if not arg1_attn_mask:
      arg1_attn_mask = []

      for arg1_ids in arg1:
        arg1_mask = []
        for arg1_id in arg1_ids:
          if arg1_id == 0: # PAD token id: 0
            arg1_mask.append(0)
          else:
            arg1_mask.append(1)
        arg1_attn_mask.append(arg1_mask)

    if not arg2_attn_mask:
      arg2_attn_mask = []

      for arg2_ids in arg2:
        arg2_mask = []
        for arg2_id in arg2_ids:
          if arg2_id == 0: # PAD token id: 0
            arg2_mask.append(0)
          else:
            arg2_mask.append(1)
        arg2_attn_mask.append(arg2_mask)
    ############################################################################

    ops = []
    for fetch_op in fetch_ops:
      if fetch_op in self.fetch_ops_dict:
        ops.append(self.fetch_ops_dict[fetch_op])
      else:
        raise ValueError(
          f"`fetch_ops` contains a `fetch_op` name {fetch_op} that is not"
          " understood")

    if self.split_args_in_embedding:
      src_batch_size = len(arg1)
      tgt_batch_size = src_batch_size * 2

      seq_length = len(arg1[0])

      arg = np.zeros([tgt_batch_size, seq_length], dtype=np.float32)
      arg[0::2] = arg1
      arg[1::2] = arg2

      arg_attn_mask = np.zeros([tgt_batch_size, seq_length], dtype=np.float32)
      arg_attn_mask[0::2] = arg1_attn_mask
      arg_attn_mask[1::2] = arg2_attn_mask

      feed_dict = {
        self.arg: arg,
        self.conn: conn,
        self.label: label_ids,
        self.arg_attn_mask: arg_attn_mask
      }
    else:
      feed_dict = {
        self.arg1          : arg1,
        self.arg2          : arg2,
        self.conn          : conn,
        self.label         : label_ids,
        self.arg1_attn_mask: arg1_attn_mask,
        self.arg2_attn_mask: arg2_attn_mask
      }

    return ops, feed_dict

  def postprocess_batch_vals(self, batch, values, fetch_ops,
                             l2i_mapping=None,
                             exid_to_feature_mapping=None):
    # tedious decoupling
    label_ids = []
    batch_bert_outputs = []
    arg1_mask = []
    arg2_mask = []

    # TODO (May 5): should move this to BERTEmbedding, as `convert_to_vals` or -
    #   at least part of it.
    for batch_example in batch:
      # exid indexes into values to fetch correct values
      batch_exid = batch_example.exid

      batch_bert_outputs.append(values[batch_exid])

      # label
      label_ids.append(l2i_mapping(batch_example.label))

      batch_feature = exid_to_feature_mapping[batch_exid]

      batch_arg1_feature = batch_feature[0]
      batch_arg1_mask = batch_arg1_feature.input_mask
      arg1_mask.append(batch_arg1_mask)

      batch_arg2_feature = batch_feature[1]
      batch_arg2_mask = batch_arg2_feature.input_mask
      arg2_mask.append(batch_arg2_mask)

    # prepare bert output: [batch, total_seq_length, bert_hidden_size]
    batch_bert_outputs = np.asarray(batch_bert_outputs)
    total_seq_length = batch_bert_outputs.shape[1]
    assert total_seq_length == self.max_arg_length * 2, \
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
      self.arg1_attn_mask: arg1_mask,
      self.arg2_attn_mask: arg2_mask
    }

    return feed_dict