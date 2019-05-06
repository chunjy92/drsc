#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

__author__ = 'Jayeol Chun'


class MLP(object):
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
               do_finetune_embedding=False,
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
    self.do_finetune_embedding = do_finetune_embedding

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

  def init_embedding(self, placeholder):
    embedding_table = tf.get_variable(
      name="embedding_table",
      shape=self.embedding_shape,
      trainable=self.do_finetune_embedding
    )

    self.embedding_init_op = embedding_table.assign(placeholder)
    return embedding_table


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

  def build_input_pipeline(self):
    if self.embedding == "bert":
      self.arg1 = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="arg1")
      self.arg2 = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="arg2")
      # TODO: max_len for conn???
      self.conn = tf.placeholder(
        tf.float32, [None, self.max_arg_length, self.word_vector_width],
        name="conn")
      self.label = tf.placeholder(tf.int32, [None], name="label")

      # not used but only for compatibility
      # placehodlers for attention_mask
      self.arg1_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg1_attention_mask")
      self.arg2_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg2_attention_mask")

      # for consistency
      self.embedding_placeholder = None
      self.embedding_table = None

      return self.arg1, self.arg2
    else:
      self.arg1 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="arg1")
      self.arg2 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="arg2")
      # TODO: max_len for conn???
      self.conn = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="conn")
      self.label = tf.placeholder(tf.int32, [None], name="label")

      self.embedding_placeholder = \
        tf.placeholder(tf.float32, self.embedding_shape,
                       "embedding_placeholder")

      self.embedding_table = self.init_embedding(self.embedding_placeholder)

      # embedding lookup
      with tf.variable_scope("embedding"):
        arg1 = tf.nn.embedding_lookup(self.embedding_table, self.arg1)
        arg2 = tf.nn.embedding_lookup(self.embedding_table, self.arg2)

      return arg1, arg2

  def build(self, scope=None):
    with tf.variable_scope(scope, default_name="mlp_model"):
      arg1, arg2 = self.build_input_pipeline()

      if self.sense_type == "implicit":
        if self.do_pooling_first:
          with tf.variable_scope("pooling"):
            arg1_pooled = apply_pooling_fn(arg1,
                                           pooling_action=self.pooling_action)
            arg2_pooled = apply_pooling_fn(arg2,
                                           pooling_action=self.pooling_action)

          combined = self.combine_pooled_tensors(arg1_pooled, arg2_pooled,
                                                   add_bias=True)

          output = self.build_dense_layers_single_input(combined)

        else:
          raise NotImplementedError("")
      else:
        # TODO: for other sense types
        raise NotImplementedError()


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
        labels=one_hot_labels,
        logits=logits)
      loss = tf.reduce_mean(self.per_example_loss)

      reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
      reg_constant = 0.01
      self.loss = loss + reg_constant * tf.reduce_sum(reg_losses)

    optimizer = get_optimizer(self.optimizer)
    self.train_op = optimizer(self.learning_rate).minimize(self.loss)

def apply_pooling_fn(input_tensor, second_tensor=None, pooling_action=None):
  # tensor shape: [batch, arg_length, word_vector_width]
  if pooling_action == "sum":
    if second_tensor:
      return tf.add(input_tensor, second_tensor)
    return tf.reduce_sum(input_tensor, axis=1)
  elif pooling_action== "mean":
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

def get_optimizer(optimizer):
  if optimizer=="adam":
    return tf.train.AdamOptimizer
  elif optimizer == "sgd":
    return tf.train.GradientDescentOptimizer
  elif optimizer == 'adagrad':
    return tf.train.AdagradOptimizer
  else:
    raise NotImplementedError()
