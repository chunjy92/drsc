#! /usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import tqdm

from bert import modeling, tokenization

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


class BERTEmbedding(object):
  def __init__(self,
               model_dir,
               bert_config_file,
               vocab_file,
               init_checkpoint,
               batch_size=32,
               max_arg_length=128,
               truncation_mode="normal",
               do_lower_case=False,
               use_one_hot_embeddings=False):
    self.model_dir = model_dir
    self.bert_config_file = bert_config_file
    self.vocab_file = vocab_file
    self.init_checkpoint = init_checkpoint
    self.batch_size = batch_size
    self.max_arg_length = max_arg_length
    self.truncation_mode = truncation_mode
    self.do_lower_case = do_lower_case
    self.use_one_hot_embeddings = use_one_hot_embeddings

    # Word-Piece tokenizer
    self.tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

    # load bert
    tokenization.validate_case_matches_checkpoint(self.do_lower_case,
                                                  self.init_checkpoint)
    self.bert_config = \
      modeling.BertConfig.from_json_file(self.bert_config_file)

    # max_position_embeddings==512
    if self.max_arg_length > self.bert_config.max_position_embeddings:
      raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (self.max_arg_length, self.bert_config.max_position_embeddings))

  def run(self, examples):
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

    return bert_outputs, exid_to_feature


  def get_embedding_table(self):
    raise ValueError(
      "BERT Embedding produces embedding table by `run` for each data example")

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

    # tf.logging.info("**** Trainable Variables ****")
    # for var in tvars:
    #   init_string = ""
    #   if var.name in initialized_variable_names:
    #     init_string = ", *INIT_FROM_CKPT*"
    #   tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
    #                   init_string)

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
