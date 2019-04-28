#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tqdm import trange

__author__ = 'Jayeol Chun'


class MLP(object):
  def __init__(self,
               labels=None,
               max_arg_length=128,
               feature_size=50,
               hidden_size=100,
               num_hidden_layers=2,
               learning_rate=0.0001,
               optimizer='adam',
               dataset_type='implicit',
               pooling_action='sum_first',
               conn_action=None,
               embedding_shape=None,
               finetune_embedding=False):
    # model architecture will slightly vary depending on combinations of:
    # [dataset_type, pooling_action, conn_action]
    self.dataset_type = dataset_type # see const.DATASET_TYPES
    self.pooling_action = pooling_action # see const.POOLING_ACTIONS
    pooling_action_split = pooling_action.split("_")
    self.pooling_action = pooling_action_split[0] # sum, mean, concat, ..
    self.pooling_timing = pooling_action_split[1] # first or later

    self.conn_action = conn_action # see const.CONN_ACTIONS

    # experimental settings
    self.max_arg_length = max_arg_length
    self.feature_size = feature_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.learning_rate = learning_rate
    self.optimizer = optimizer

    # data-related
    self.labels = labels
    self.num_labels = len(self.labels)

    # embedding related
    self.embedding_shape = embedding_shape
    self.finetune_embedding = finetune_embedding

    self.build()

  def build_dense_layers_single_input(self, input, num_layers=None):
    num_layers = num_layers if num_layers else self.num_hidden_layers

    output = input

    for i in range(num_layers):
      with tf.variable_scope(f"layer_{i}"):
        layer_input = output

        if i == num_layers-1:
          hidden_size = self.num_labels
          activation = None
        else:
          hidden_size = self.hidden_size
          activation = tf.nn.relu

        tf.logging.info(f"Hidden Size: {hidden_size}")
        output = tf.layers.dense(
          name="dense",
          inputs=layer_input,
          units=hidden_size,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
          activation=activation
        )

    return output

  def init_embedding(self, placeholder):
    embedding_table = tf.get_variable(
      name="embedding",
      shape=self.embedding_shape,
      trainable=self.finetune_embedding
    )
    self.embedding_init_op = embedding_table.assign(placeholder)

    return embedding_table


  def combine_tensors(self, input_1, input_2, target_hidden_size=None):
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
    input_matmul = tf.add(input_1_matmul, input_2_matmul)

    bias = tf.get_variable(
      f"combine_bias", [target_hidden_size], initializer=tf.zeros_initializer()
    )

    return tf.nn.bias_add(input_matmul, bias)

  def build(self):
    with tf.variable_scope("mlp"):
      self.arg1 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="arg1")
      self.arg2 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="arg2")
      self.label = tf.placeholder(tf.int32, [None, self.num_labels],
                                  name="label")
      # TODO: max_len for conn???
      self.conn = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="conn")

      with tf.variable_scope("embedding"):
        self.embedding_placeholder = \
          tf.placeholder(tf.float32, self.embedding_shape,
                         "embedding_placeholder")
        self.embedding_table = self.init_embedding(self.embedding_placeholder)

      # embedding lookup
      with tf.variable_scope("embedding"):
        # arg1_embedding = self.embedding_lookup(self.arg1)
        arg1_embedding = tf.nn.embedding_lookup(
          self.embedding_table, self.arg1)
        # arg2_embedding = self.embedding_lookup(self.arg2)
        arg2_embedding = tf.nn.embedding_lookup(
          self.embedding_table, self.arg2)

      if self.dataset_type == "implicit":
        if self.pooling_timing == "first":
          with tf.variable_scope("pooling"):
            pooling_fn = get_pooling_fn(self.pooling_action)
            arg1_pooled = pooling_fn(arg1_embedding, axis=1)
            arg2_pooled = pooling_fn(arg2_embedding, axis=1)
            combined = self.combine_tensors(arg1_pooled, arg2_pooled)

          output = self.build_dense_layers_single_input(combined)

      else:
        pass

    with tf.variable_scope("loss"):
      logits = output

      tf.logging.info(f"Logits Shape: {logits.shape}")

      self.preds = tf.argmax(logits, axis=1)
      golds = tf.argmax(self.label, axis=1)
      self.acc = tf.reduce_mean(tf.cast(tf.equal(self.preds, golds), "float"),
                                name="accuracy")

      self.probabilities = tf.nn.softmax(logits, axis=-1)
      self.per_example_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=self.label,
        logits=logits
      )
      self.loss = tf.reduce_mean(self.per_example_loss)

    optimizer = get_optimizer(self.optimizer)
    self.train_op = optimizer(self.learning_rate).minimize(self.loss)

  # def train(self, sess, examples):
  #
  #
  #
  #     _, loss = self.sess.run([self.train_op, self.loss],
  #                             feed_dict={self.x: train_x, self.y: train_y})


def get_pooling_fn(pooling):
  if pooling == "sum":
    return tf.reduce_sum
  elif pooling=="mean":
    return tf.reduce_mean
  elif pooling == "max":
    return tf.reduce_max
  # elif .pooling == "matmul":
  #   return tf.matmul
  else:
    raise ValueError("pooling function not understood")

def get_optimizer(optimizer):
  if optimizer=="adam":
    return tf.train.AdamOptimizer
  elif optimizer == "sgd":
    return tf.train.GradientDescentOptimizer
  else:
    raise NotImplementedError()
