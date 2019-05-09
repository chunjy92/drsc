#! /usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import pickle

import numpy as np
import tensorflow as tf
import tqdm

from bert import modeling, tokenization
from utils import const
from .Embedding import Embedding

__author__ = 'Jayeol Chun'


class BERTInputFeatures(object):
  def __init__(self,
               exid,
               input_ids,
               segment_ids,
               input_mask,
               arg_type):
    self.exid = exid
    self.input_ids = input_ids
    self.segment_ids = segment_ids
    self.input_mask = input_mask
    self.arg_type = arg_type


class BERTEmbedding(Embedding):
  def __init__(self,
               model_dir,
               bert_config_file,
               vocab_file,
               init_checkpoint,
               batch_size=32,
               max_arg_length=128,
               truncation_mode="normal",
               do_lower_case=False,
               finetune_embedding=False,
               is_training=False,
               padding_action='normal',
               use_one_hot_embeddings=False,
               scope=None):
    self.model_dir = model_dir
    self.bert_config_file = bert_config_file
    self.vocab_file = vocab_file
    self.init_checkpoint = init_checkpoint
    self.batch_size = batch_size
    self.max_arg_length = max_arg_length
    self.truncation_mode = truncation_mode
    self.do_lower_case = do_lower_case
    self.finetune_embedding = finetune_embedding
    self.is_training = is_training
    self.padding_action = padding_action
    self.use_one_hot_embeddings = use_one_hot_embeddings

    # Word-Piece tokenizer
    self.tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

    # load bert
    tokenization.validate_case_matches_checkpoint(self.do_lower_case,
                                                  self.init_checkpoint)
    self.bert_config = copy.deepcopy(
      modeling.BertConfig.from_json_file(self.bert_config_file))



    self._embedding_table = None
    self._vocab = tokenization.load_vocab(self.vocab_file)

    # max_position_embeddings==512
    if self.max_arg_length > self.bert_config.max_position_embeddings:
      raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (self.max_arg_length, self.bert_config.max_position_embeddings))

    if self.finetune_embedding:
      self.build(scope)

  ################################### BUILD ####################################
  def build_bert_model(self,
                       input_ids,
                       input_mask,
                       token_type_ids):
    if not self.is_training:
      self.bert_config.hidden_dropout_prob = 0.0
      self.bert_config.attention_probs_dropout_prob = 0.0

    with tf.variable_scope('bert'):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (embedding_output, _) = modeling.embedding_lookup(
            input_ids=input_ids,
            vocab_size=self.bert_config.vocab_size,
            embedding_size=self.bert_config.hidden_size,
            initializer_range=self.bert_config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=self.use_one_hot_embeddings
        )

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        embedding_output = modeling.embedding_postprocessor(
            input_tensor=embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=self.bert_config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=self.bert_config.initializer_range,
            max_position_embeddings=self.bert_config.max_position_embeddings,
            dropout_prob=self.bert_config.hidden_dropout_prob
        )

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = modeling.create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer
        self.all_encoder_layers = modeling.transformer_model(
            input_tensor=embedding_output,
            attention_mask=attention_mask,
            hidden_size=self.bert_config.hidden_size,
            num_hidden_layers=self.bert_config.num_hidden_layers,
            num_attention_heads=self.bert_config.num_attention_heads,
            intermediate_size=self.bert_config.intermediate_size,
            intermediate_act_fn=modeling.get_activation(
              self.bert_config.hidden_act),
            hidden_dropout_prob=self.bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=\
              self.bert_config.attention_probs_dropout_prob,
            initializer_range=self.bert_config.initializer_range,
            do_return_all_layers=True
        )

        # `final_layer` shape = [batch_size, seq_length, hidden_size]
        self.sequence_output = self.all_encoder_layers[-1]

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def build(self, scope=None):

    with tf.variable_scope(scope, default_name="bert_embedding_model"):
      self.arg1 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="arg1")
      self.arg2 = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="arg2")
      self.label = tf.placeholder(tf.int32, [None], name="label")

      # TODO: max_len for conn???
      self.conn = tf.placeholder(tf.int32, [None, self.max_arg_length],
                                 name="conn")

      # placehodlers for attention_mask
      self.arg1_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg1_attention_mask")
      self.arg2_attn_mask = tf.placeholder(
        tf.int32, [None, self.max_arg_length], name="arg2_attention_mask")

    arg_concat = tf.concat([self.arg1, self.arg2], axis=1)
    self.bert_mask_concat = \
      tf.concat([self.arg1_attn_mask, self.arg2_attn_mask], axis=1)

    segment_ids = tf.concat([
      tf.zeros_like(self.arg1, dtype=tf.int32),  # Arg1: 0s
      tf.ones_like(self.arg2, dtype=tf.int32)  # Arg2: 1s
    ], axis=1)

    self.build_bert_model(
      arg_concat, input_mask=self.bert_mask_concat, token_type_ids=segment_ids)

    self.bert_arg_concat = self.get_sequence_output()

    ########################### Finetunable BERT #################################
  def convert_to_ids(self, examples, l2i, **kwargs):
    arg1, arg2, conn, labels, arg1_mask, arg2_mask = [], [], [], [], [], []

    for example in examples:
      tokens_a = self.tokenizer.tokenize(example.arg1)
      tokens_b = self.tokenizer.tokenize(example.arg2)

      if len(tokens_a) > self.max_arg_length - 2:
        tokens_a = tokens_a[0:(self.max_arg_length - 2)]
      tokens_a.insert(0, "[CLS]")
      tokens_a.append("[SEP]")

      if len(tokens_b) > self.max_arg_length - 1:
        tokens_b = tokens_b[0:(self.max_arg_length - 1)]
      tokens_b.append("[SEP]")

      input_ids_a = self.tokenizer.convert_tokens_to_ids(tokens_a)
      input_ids_b = self.tokenizer.convert_tokens_to_ids(tokens_b)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask_a = [1] * len(input_ids_a)
      while len(input_ids_a) < self.max_arg_length:
        if self.padding_action == 'pad_left_arg1':
          input_ids_a.insert(1, 0)
          input_mask_a.insert(1, 0)
        else:
          input_ids_a.append(0)
          input_mask_a.append(0)

      input_mask_b = [1] * len(input_ids_b)
      while len(input_ids_b) < self.max_arg_length:
        input_ids_b.append(0)
        input_mask_b.append(0)

      label_id = l2i(example.label)

      # into data
      arg1.append(input_ids_a)
      arg2.append(input_ids_b)
      labels.append(label_id)
      # TODO: Connectives jsut padding values
      conn.append([self.vocab[const.PAD]] * self.max_arg_length)
      arg1_mask.append(input_mask_a)
      arg2_mask.append(input_mask_b)

    return arg1, arg2, conn, labels, arg1_mask, arg2_mask

  ############################ STATIC BERT ONLY ################################
  def convert_to_values(self, examples, filename=None):
    if filename:
      with open(filename, 'rb') as f:
        return pickle.load(f)

    features = convert_examples_to_embedding_features(
      examples, seq_length=self.max_arg_length, tokenizer=self.tokenizer)

    # maps `example.exid` to feature used in BERTEmbedding
    exid_to_feature = {}
    for feature in features:
      exid = feature.exid

      if exid not in exid_to_feature:
        exid_to_feature[exid] = [feature]
      else:
        exid_to_feature[exid].insert(feature.arg_type, feature)

    model_fn = model_fn_builder(
      bert_config=self.bert_config,
      init_checkpoint=self.init_checkpoint,
      use_one_hot_embeddings=self.use_one_hot_embeddings
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    runConfig = tf.estimator.RunConfig(session_config=config)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=self.model_dir,
                                       config=runConfig)

    input_fn = input_fn_builder(
      features=features, seq_length=self.max_arg_length,
      batch_size=self.batch_size)

    # for bert_output data
    bert_outputs = np.zeros(
      [len(examples), self.max_arg_length*2, self.bert_config.hidden_size],
      dtype=np.float)

    tf.logging.info("Begin running BERT")
    for result in tqdm.tqdm(estimator.predict(input_fn,
                                              yield_single_examples=True)):
      exid = result['exid']
      arg_type = result['arg_type']
      data = np.array(result['last_layer_output'])

      if arg_type == 0:
        bert_outputs[exid][:self.max_arg_length] = data
      else:
        bert_outputs[exid][self.max_arg_length:] = data

    if filename:
      with open(filename, 'wb') as f:
        pickle.dump((bert_outputs, exid_to_feature), f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    return bert_outputs, exid_to_feature

  ################################## GETTERS ###################################
  def get_arg_concat(self):
    return self.bert_arg_concat

  def get_attn_mask(self):
    return self.bert_mask_concat

  ############################### PLACEHOLDERS #################################
  def get_arg1(self):
    return self.arg1

  def get_arg2(self):
    return self.arg2

  def get_label(self):
    return self.label

  def get_conn(self):
    return self.conn

  def get_arg1_mask(self):
    return self.arg1_attn_mask

  def get_arg2_mask(self):
    return self.arg2_attn_mask

  def get_all_placeholder_ops(self):
    return self.arg1, self.arg2, self.conn, self.label, \
           self.arg1_attn_mask, self.arg2_attn_mask

  ################################# POSTPROCESS ################################
  def postprocess_batch_ids(self, batch):
    arg1, arg2, conn, label_ids, arg1_mask, arg2_mask = batch

    arg1_attn_mask = arg1_mask
    arg2_attn_mask = arg2_mask

    # arg1 attn mask
    if not arg1_attn_mask:
      arg1_attn_mask = []

      for arg1_ids in arg1:
        arg1_mask = []
        for arg1_id in arg1_ids:
          if arg1_id == self.vocab[const.PAD]:
            arg1_mask.append(0)
          else:
            arg1_mask.append(1)
        arg1_attn_mask.append(arg1_mask)

    # arg2 attn mask
    if not arg2_attn_mask:
      arg1_attn_mask = []

      for arg2_ids in arg2:
        arg2_mask = []
        for arg2_id in arg2_ids:
          if arg2_id == self.vocab[const.PAD]:
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

    return feed_dict

def convert_examples_to_embedding_features(examples, seq_length, tokenizer):
  """Loads a data file into a list of `InputBatch`s."""

  features = []
  for (ex_index, example) in enumerate(examples):
    tokens_a = tokenizer.tokenize(example.arg1)
    tokens_b = tokenizer.tokenize(example.arg2)

    if len(tokens_a) > seq_length - 2:
      tokens_a = tokens_a[0:(seq_length - 2)]
    tokens_a.insert(0, "[CLS]")
    tokens_a.append("[SEP]")

    if len(tokens_b) > seq_length - 2:
      tokens_b = tokens_b[0:(seq_length - 2)]
    tokens_b.insert(0, "[CLS]")
    tokens_b.append("[SEP]")

    input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
    input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask_a = [1] * len(input_ids_a)
    while len(input_ids_a) < seq_length:
      input_ids_a.append(0)
      input_mask_a.append(0)

    input_mask_b = [1] * len(input_ids_b)
    while len(input_ids_b) < seq_length:
      input_ids_b.append(0)
      input_mask_b.append(0)

    input_type_ids_a = [0] * seq_length
    input_type_ids_b = [0] * seq_length

    assert len(input_ids_a) == seq_length
    assert len(input_ids_b) == seq_length
    assert len(input_mask_a) == seq_length
    assert len(input_mask_b) == seq_length
    assert len(input_type_ids_a) == seq_length
    assert len(input_type_ids_b) == seq_length

    exid = example.exid

    features.extend(
        [BERTInputFeatures(
          exid=exid,
          input_ids=input_ids_a,
          segment_ids=input_type_ids_a,
          input_mask=input_mask_a,
          arg_type=0
        ),
          BERTInputFeatures(
            exid=exid,
            input_ids=input_ids_b,
            segment_ids=input_type_ids_b,
            input_mask=input_mask_b,
            arg_type=1
        )]
    )
  return features

############################# STATIC BERT ONLY #################################
def model_fn_builder(bert_config, init_checkpoint, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    exid = features['exid']
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    arg_type = features["arg_type"]

    model = modeling.BertModel(
        config=bert_config,
        is_training=False,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode != tf.estimator.ModeKeys.PREDICT:
      raise ValueError("Only PREDICT modes are supported: %s" % (mode))

    tvars = tf.trainable_variables()
    (assignment_map,
     initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
         tvars, init_checkpoint)

    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    last_layer_output = model.get_sequence_output()

    predictions = {
      "exid": exid,
      "arg_type": arg_type,
      "last_layer_output": last_layer_output
    }

    output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    return output_spec

  return model_fn

def input_fn_builder(features, seq_length, batch_size):
  """Creates an `input_fn` closure to be passed to Estimator."""

  all_exids = []
  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_arg_types = []

  for feature in features:
    all_exids.append(feature.exid)
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_arg_types.append(feature.arg_type)

  def input_fn():
    """The actual input function."""

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "exid":
            tf.constant(
                all_exids,
                shape=[num_examples],
                dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "arg_type":
            tf.constant(
                all_arg_types,
                shape=[num_examples],
                dtype=tf.int32),

    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn
