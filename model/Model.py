#! /usr/bin/python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

import tensorflow as tf

from bert import optimization
from utils import const

__author__ = 'Jayeol Chun'


class Model(ABC):
  def __init__(self,
               labels=None,
               attention_type=None,
               max_arg_length=128,
               word_vector_width=768,
               hidden_size=768,
               num_hidden_layers=3,
               num_attention_heads=8,
               hidden_dropout_prob=0.1,
               learning_rate=3e-4,
               optimizer='adam',
               num_train_steps=0,
               num_warmup_steps=0,
               embedding=None,
               embedding_name=None,
               embedding_shape=None,
               split_args_in_embedding=False,
               is_training=False,
               sense_type='implicit',
               pooling_action='concat',
               cls_action='first_cls',
               conn_action=None,
               do_pooling_first=False,
               finetune_embedding=False,
               scope=None):
    # model architecture will slightly vary depending on combinations of:
    # [dataset_type, pooling_action, conn_action]
    self.attention_type = attention_type
    self.sense_type = sense_type  # see const.DATASET_TYPES
    self.pooling_action = pooling_action  # see const.POOLING_ACTIONS
    self.split_args_in_embedding = split_args_in_embedding
    self.embedding = embedding
    self.embedding_name = embedding_name
    self.embedding_shape = embedding_shape

    # model settings
    self.max_arg_length = max_arg_length
    self.word_vector_width = word_vector_width
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_dropout_prob = hidden_dropout_prob
    self.is_training = is_training

    # optimizer
    self.learning_rate = learning_rate
    self.optimizer_name = optimizer
    # self.optimizer = self.get_optimizer(self.optimizer_name)
    self.num_train_steps = num_train_steps
    self.num_warmup_steps = num_warmup_steps

    # data-related
    self.labels = labels
    self.num_labels = len(self.labels)

    # control flags
    self.pooling_action = pooling_action
    self.do_pooling_first = do_pooling_first
    self.finetune_embedding = finetune_embedding
    self.cls_action = cls_action
    self.conn_action = conn_action  # see const.CONN_ACTIONS

    # boolean flags for embedding type
    self.is_bert_embedding = self.embedding_name == 'bert'
    self.is_finetunable_bert_embedding = \
      self.is_bert_embedding and self.finetune_embedding

    self.build(scope)

    self.compile_fetch_ops()

  ################################### BUILD ####################################
  @abstractmethod
  def build(self, scope=None):
    pass

  def build_loss_op(self, output):
    # loss function
    hidden_size = output.shape[-1].value

    output_weights = tf.get_variable(
      "output_weights", [self.num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      if self.is_training:
        # I.e., 0.1 dropout
        output = tf.nn.dropout(output, keep_prob=0.9)

      logits = tf.matmul(output, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)

      self.per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.label, logits=logits, name="per_loss")
      self.loss = tf.reduce_mean(self.per_example_loss, name="mean_loss")

    return logits

  def build_train_op(self):
    if self.is_bert_embedding and not self.finetune_embedding:
      # remove pre-trained bert parameters from `TRAINABLE_VARIABLES`
      var_list = tf.trainable_variables()
      tf.get_default_graph().clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

      # trainable_vars_no_bert = []
      for var in var_list:
        if not var.name.startswith("bert"):
          tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, var)

    # if self.optimizer_name == "adam" and self.num_warmup_steps > 0:
    #   tf.logging.info("Adam Weight Decay Optimizer")
    #   train_op = optimization.create_optimizer(
    #     loss=self.loss, init_lr=self.learning_rate,
    #     num_train_steps=self.num_train_steps,
    #     num_warmup_steps=self.num_warmup_steps,
    #     use_tpu=False
    #   )
    # else:
    optimizer = self.get_optimizer(self.optimizer_name)
    tf.logging.info(f"{self.optimizer_name.capitalize()} Optimizer")
    train_op = optimizer(self.learning_rate).minimize(self.loss,
                                                      name='train_op')
    return train_op

  ############################### POSTPROCESS ##################################
  @abstractmethod
  def postprocess_batch(self, batch, fetch_ops):
    pass

  def fetch_ops(self, fetch_op_names):
    ops = []
    for fetch_op in fetch_op_names:
      if fetch_op in self.fetch_ops_dict:
        ops.append(self.fetch_ops_dict[fetch_op])
      else:
        raise ValueError(
          f"`fetch_ops` contains a `fetch_op` name {fetch_op} that is not"
          " understood")
    return ops

  ############################## PLACEHOLDER OPS ###############################
  def init_embedding(self, placeholder):

    embedding_table = tf.get_variable(
      name="embedding_table",
      shape=self.embedding_shape,
      trainable=self.finetune_embedding
    )

    self.embedding_init_op = embedding_table.assign(placeholder)

    return embedding_table

  def build_input_pipeline(self):
    """Build all placeholder ops"""

    if self.is_bert_embedding:
      placeholer_ops = self.embedding.get_all_placeholder_ops()

      if self.split_args_in_embedding:
        self.arg = placeholer_ops[0]
        self.conn = placeholer_ops[1]
        self.label = placeholer_ops[2]
        self.arg_attn_mask = placeholer_ops[3]
      else:
        self.arg1 = placeholer_ops[0]
        self.arg2 = placeholer_ops[1]
        self.conn = placeholer_ops[2]
        self.label = placeholer_ops[3]
        self.arg1_attn_mask = placeholer_ops[4]
        self.arg2_attn_mask = placeholer_ops[5]

      self.embedding_placeholder = None
      self.embedding_table = None

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

      self.arg1_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg1_attn_mask")
      self.arg2_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg2_attn_mask")

  ################################### UTIL #####################################
  def compile_fetch_ops(self):
    self.fetch_ops_dict = {
      "per_loss": self.per_example_loss,
      "mean_loss": self.loss,
      "preds": self.preds,
      "correct": self.correct,
      "acc": self.acc,
      "train_op": self.train_op
    }

  def apply_cls_pooling_fn(self, input_tensor, cls_action=None):
    first_cls = tf.squeeze(input_tensor[:, 0:1, :], axis=1)
    second_cls = tf.squeeze(
      input_tensor[:, self.max_arg_length:self.max_arg_length+1, :], axis=1)

    if not cls_action:
      cls_action = self.cls_action

    pooled_tensor = None
    if cls_action == "first_cls":
      pooled_tensor = first_cls
    elif cls_action == "second_cls":
      pooled_tensor = second_cls
    elif cls_action == 'concat':
      pooled_tensor = tf.concat([first_cls, second_cls], axis=-1)
    elif cls_action == 'new_cls':
      # TODO: requires re-structuring
      raise NotImplementedError("Currently `new_cls` is not supported")
    else:
      if cls_action in const.POOLING_ACTIONS:
        pooled_tensor = self.apply_pooling_fn(input_tensor,
                                              pooling_action=cls_action)
      else:
        raise ValueError("CLS Pooling action not understood")

    return pooled_tensor

  def apply_pooling_fn(self, input_tensor, second_tensor=None,
                       pooling_action=None):
    # tensor shape: [batch, arg_length, word_vector_width]
    if not pooling_action:
      pooling_action = self.pooling_action

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
      raise ValueError("Pooling function not understood")

  def get_optimizer(self, optimizer):
    if optimizer == "adam":
      return tf.train.AdamOptimizer
    elif optimizer == "sgd":
      return tf.train.GradientDescentOptimizer
    elif optimizer == 'adagrad': # Te's optimizer
      return tf.train.AdagradOptimizer
    else:
      raise ValueError("Optimizer value not understood")
