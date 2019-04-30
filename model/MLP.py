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
    self.embedding_shape = embedding_shape
    self.do_finetune_embedding = do_finetune_embedding

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
          activation=tf.nn.tanh
        )

    return output

  def init_embedding(self, placeholder):
    embedding_table = tf.get_variable(
      name="embedding",
      shape=self.embedding_shape,
      trainable=self.do_finetune_embedding
    )

    self.embedding_init_op = embedding_table.assign(placeholder)

    return embedding_table


  def linearly_combine_tensors(self, input_1, input_2, target_hidden_size=None,
                               add_bias=False):
    hidden_size = input_1.shape[-1].value
    target_hidden_size = \
      target_hidden_size if target_hidden_size else self.hidden_size

    weight_1 = tf.get_variable(
      f"combine_weight_{1}", [target_hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02)
    )

    weight_2 = tf.get_variable(
      f"combine_weight_{2}", [target_hidden_size, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02)
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
        arg1_embedding = tf.nn.embedding_lookup(self.embedding_table, self.arg1)
        arg2_embedding = tf.nn.embedding_lookup(self.embedding_table, self.arg2)

        # conn_embedding = tf.nn.embedding_lookup(self.embedding_table, self.conn)

      if self.sense_type == "implicit":
        if self.do_pooling_first:
          with tf.variable_scope("pooling"):
            arg1_pooled = apply_pooling_fn(arg1_embedding,
                                           pooling_action=self.pooling_action)
            arg2_pooled = apply_pooling_fn(arg2_embedding,
                                           pooling_action=self.pooling_action)

          combined = self.linearly_combine_tensors(arg1_pooled, arg2_pooled,
                                                   add_bias=True)

          output = self.build_dense_layers_single_input(combined)

        else:
          raise NotImplementedError(
            "Pooling is always applied first on word vectors")
      else:
        # TODO: for other sense types
        raise NotImplementedError()


    hidden_size = output.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [self.num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      logits = tf.matmul(output, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
      self.preds = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

      one_hot_labels = tf.one_hot(self.label, depth=self.num_labels,
                                  dtype=tf.float32)

      self.acc = tf.reduce_mean(
        tf.cast(tf.equal(self.preds, self.label), "float"), name="accuracy")

      self.per_example_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=one_hot_labels,
        logits=logits
      )
      self.loss = tf.reduce_mean(self.per_example_loss)

      # probabilities = tf.nn.softmax(logits, axis=-1)
      # log_probs = tf.nn.log_softmax(logits, axis=-1)
      #
      # per_example_loss = -tf.reduce_sum(tf.cast(self.label, tf.float32) * log_probs, axis=-1)
      # self.loss = tf.reduce_mean(per_example_loss)

    # with tf.variable_scope("loss"):
    #   logits = output
    #
    #   tf.logging.info(f"Logits Shape: {logits.shape}")
    #
    #   self.preds = tf.argmax(logits, axis=1)
    #   golds = tf.argmax(self.label, axis=1)
    #   self.acc = tf.reduce_mean(tf.cast(tf.equal(self.preds, golds), "float"),
    #                             name="accuracy")
    #
    #   self.probabilities = tf.nn.softmax(logits, axis=-1)
    #
    #   log_probs = tf.nn.log_softmax(logits, axis=-1)
    #   per_example_loss = -tf.reduce_sum(tf.cast(self.label, tf.float32) * log_probs, axis=-1)
    #   self.loss = tf.reduce_mean(per_example_loss)

    optimizer = get_optimizer(self.optimizer)
    self.train_op = optimizer(self.learning_rate).minimize(self.loss)

# def apply_join_fn(input_tensor, second_tensor, join_action=None):
#   if join_action == "sum":
#     if second_tensor:
#       return tf.add(input_tensor, second_tensor)
#     return tf.reduce_sum(input_tensor, axis=1)
#   elif join_action == "mean":
#     if second_tensor:
#       return tf.reduce_mean([input_tensor, second_tensor], axis=0)
#     return tf.reduce_mean(input_tensor, axis=1)
#   elif join_action == "max":
#     if second_tensor:
#       return tf.reduce_max([input_tensor, second_tensor], axis=0)
#     return tf.reduce_max(input_tensor, axis=1)
#   elif join_action in ["concat", 'matmul']:
#     # usually works on model outputs for each arg1 and arg2
#     if not second_tensor:
#       raise ValueError("Second tensor passed as `None` value")
#     input_tensor_shape = input_tensor.shape
#     second_tensor_shape = second_tensor.shape
#     assert_op = tf.assert_equal(input_tensor_shape, second_tensor_shape)
#     with tf.control_dependencies([assert_op]):
#       if join_action == 'concat':
#         return tf.concat([input_tensor, second_tensor], axis=-1)
#       else:
#         return tf.multiply(input_tensor, second_tensor)
#
#   else:
#     raise ValueError(f"{join_action} pooling function not understood")


def apply_pooling_fn(input_tensor, second_tensor=None, pooling_action=None):
  # tensor shape: [batch, arg_length, word_vector_width]
  tf.logging.info(input_tensor.shape)
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
