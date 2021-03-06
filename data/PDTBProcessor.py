#! /usr/bin/python3
# -*- coding: utf-8 -*-
import codecs
import json
import os

import tensorflow as tf

from utils import const
from .PDTBRelation import PDTBRelation

__author__ = 'Jayeol Chun'


class PDTBProcessor(object):
  """PDTB Data Processor

  Wraps each relation as `PDTBRelation` object, defined in
  `data/PDTBRelation.py`

  """
  def __init__(self,
               data_dir=const.CONLL,
               max_seq_length=128,
               truncation_mode='normal',
               do_lower_case=False,
               sense_type='implicit',
               sense_level=2,
               multiple_senses_action="pick_first",
               padding_action="normal",
               drop_partial_data=False):
    # root data dir
    self.data_dir = data_dir

    # sense config
    self.sense_type = sense_type
    self.sense_level = sense_level
    self.multiple_senses_action = multiple_senses_action
    self.padding_action = padding_action
    self.drop_partial_data = drop_partial_data

    # preprocessing config
    self.max_seq_length = max_seq_length
    self.max_arg_length = int(max_seq_length/2)
    self.truncation_mode = truncation_mode
    self.do_lower_case = do_lower_case

    # set of vocab and labels
    self._vocab = None
    self._labels = None

    # TODO (May 7): currently only assumes `implicit`
    assert self.sense_type == "implicit", \
      "Currently only supports implicit relation types"

    if self.drop_partial_data:
      self._labels = const.IMPLICIT_11_WAY
    else:
      self._labels = const.IMPLICIT_15_WAY

    # cached data
    self._cached = {}

  @property
  def vocab(self):
    return self._vocab

  @property
  def labels(self):
    return self._labels

  def remove_cache_by_key(self, key):
    if key in self._cached:
      self._cached.pop(key)

  def collect_all_vocab(self, include_blind=False):
    vocab = self.vocab
    if not vocab:
      self.compile_vocab()
      vocab = self.vocab

    vocab = set(vocab)

    dev_examples = self.get_dev_examples()
    test_examples = self.get_test_examples()
    all_examples_by_dataset = [dev_examples, test_examples]

    if include_blind:
      blind_examples = self.get_blind_examples()
      all_examples_by_dataset.append(blind_examples)

    for dataset_examples in all_examples_by_dataset:
      for example in dataset_examples:
        vocab.update(example.arg1)
        vocab.update(example.arg2)
        if example.conn:
          vocab.update(example.conn)

    return sorted(vocab)

  def compile_vocab(self, instances=None):
    """Collects unique vocab from training instances"""
    if not instances:
      tf.logging.info("Compiling vocab from training set")
      self._cached['train'] = instances = self.get_train_examples()

    vocab = set()
    for instance in instances:
      vocab.update(instance.arg1)
      vocab.update(instance.arg2)
      if instance.conn: # connective may not exist
        vocab.update(instance.conn)

    self._vocab = sorted(vocab)

  def get_train_examples(self, rel_filename=None, parse_filename=None,
                         for_bert_embedding=False):
    rel_filename = rel_filename if rel_filename else \
      "pdtb-data-01-20-15-train.json"
    parse_filename = parse_filename if parse_filename else \
      "pdtb-parses-01-12-15-train.json"
    return self._create_examples("train", rel_filename, parse_filename,
                                 for_bert_embedding=for_bert_embedding)

  def get_dev_examples(self, rel_filename=None, parse_filename=None,
                       for_bert_embedding=False):
    rel_filename = rel_filename if rel_filename else \
      "pdtb-data-01-20-15-dev.json"
    parse_filename = parse_filename if parse_filename else \
      "pdtb-parses-01-12-15-dev.json"
    return self._create_examples("dev", rel_filename, parse_filename,
                                 for_bert_embedding=for_bert_embedding)

  def get_test_examples(self, rel_filename=None, parse_filename=None,
                        for_bert_embedding=False):
    rel_filename = rel_filename if rel_filename else \
      "pdtb-data.json"
    parse_filename = parse_filename if parse_filename else \
      "pdtb-parses.json"
    return self._create_examples("test", rel_filename, parse_filename,
                                 for_bert_embedding=for_bert_embedding)

  def get_blind_examples(self, rel_filename=None, parse_filename=None,
                        for_bert_embedding=False):
    rel_filename = rel_filename if rel_filename else \
      "relations.json"
    parse_filename = parse_filename if parse_filename else \
      "parses.json"
    return self._create_examples("blind", rel_filename, parse_filename,
                                 for_bert_embedding=for_bert_embedding)

  def _create_examples(self, dataset_type, rel_filename, parse_filename,
                       for_bert_embedding=False):
    """Both Explicit and Implicit examples

    Args:
      data_dir: path to DRC dataset root
      dataset_type: all, implicit or explicit

    Returns:
      list of InputExamples
    """
    # check if cached
    if dataset_type in self._cached and not for_bert_embedding:
      res = self._cached[dataset_type]
      tf.logging.info(f"Loading {len(res)} cached {dataset_type} instances")
      return res

    dataset_dir = os.path.join(self.data_dir, dataset_type)

    # json files
    rel_filename = os.path.join(dataset_dir, rel_filename)
    parse_filename = os.path.join(dataset_dir, parse_filename)

    pdtb_f = codecs.open(rel_filename, encoding='utf-8')
    parse_f = codecs.open(parse_filename, encoding='utf8')
    parse = json.load(parse_f)

    exid = 0
    examples = []
    for i, pdtb_line in enumerate(pdtb_f):
      rel = json.loads(pdtb_line)

      # relation identifier
      doc_id = rel[const.DOC_ID]
      unique_id = rel[const.REL_ID]
      guid = f"{dataset_type}-{unique_id}"

      # relation type check
      rel_type = rel[const.TYPE].lower()
      if self.sense_type != "all":
        if self.sense_type == 'non-explicit':
          if rel_type == 'explicit':
            continue
        else:
          if self.sense_type != rel_type:
            continue

      tok_data = [None, None, None]
      tok_mask = [None, None]
      for j, token_type in enumerate([const.ARG1, const.ARG2, const.CONN]):
        # if char_span_list is empty for that token_type, no tokens exist
        # in that case, the default value of `None` is retained
        if rel[token_type][const.CHAR_SPAN_LIST]:
          token_list = rel[token_type][const.TOKEN_LIST]

          tokens = []
          for token in token_list:
            sent = parse[doc_id][const.SENTENCES][token[const.SENT_ID]]
            gold_token = sent[const.WORDS][token[const.TOK_ID]][0]

            # PTB token mapping, see `utils/const.py` for its source
            if gold_token in const.TOKEN_MAPPING:
              gold_token = const.TOKEN_MAPPING[gold_token]

            if self.do_lower_case:
              gold_token = gold_token.lower()

            tokens.append(gold_token)

          if for_bert_embedding:
            # for compatibility, since BERT expects a string. For truncation,
            # padding and masking, see `convert_to_ids` in `BERTEmbedding.py`
            tokens = " ".join(tokens)
          else:
            if self.truncation_mode == 'normal':
              tokens = tokens[:self.max_arg_length]
            else:
              tokens = tokens[-self.max_arg_length:]

            # if less than max_arg_length, pad up with _PAD_
            num_pad = self.max_arg_length - len(tokens)

            mask = []
            if self.padding_action == "pad_left_arg1" and \
                    token_type==const.ARG1:
              mask = [0] * num_pad + [1] * len(tokens)
              # pad at the beginning of tokens for arg1
              tokens = [const.PAD] * num_pad + tokens
            else:
              mask = [1] * len(tokens) + [0] * num_pad
              # pad at the end of tokens
              tokens += [const.PAD] * num_pad

            if token_type in [const.ARG1, const.ARG2]:
              tok_mask[j] = mask

          tok_data[j] = tokens

      arg1, arg2, conn = tok_data
      arg1_mask, arg2_mask = tok_mask

      sense_list = rel[const.SENSE]

      if self.multiple_senses_action == "pick_first":
        sense = sense_list[0]
        sense = to_level(sense, level=self.sense_level)

        # TODO: Only if dataset is `implicit`. Needs an update later for other
        #  dataset types or somehow make them compatible with this
        #  implementation
        if sense not in self.labels:
          continue

        input_example = PDTBRelation(
          guid=guid, exid=exid, arg1=arg1, arg2=arg2, conn=conn, label=sense,
          label_list=sense_list, arg1_mask=arg1_mask, arg2_mask=arg2_mask)
        examples.append(input_example)

        if exid < 5 and dataset_type=='train':
          tf.logging.info(input_example)

        exid += 1

      elif self.multiple_senses_action == "duplicate":
        for sense in sense_list:
          sense = to_level(sense, level=self.sense_level)

          # TODO: Only if dataset is `implicit`. Needs an update later for other
          #  dataset types or somehow make them compatible with this
          #  implementation
          if sense not in self.labels:
            continue

          input_example = PDTBRelation(
            guid=guid, exid=exid, arg1=arg1, arg2=arg2, conn=conn, label=sense,
            label_list=sense_list, arg1_mask=arg1_mask, arg2_mask=arg2_mask)
          examples.append(input_example)

          if exid < 5 and dataset_type=='train':
            tf.logging.info(input_example)

          exid += 1

      else: # smart picking by majority (?)
        raise NotImplementedError("")

    # cache
    if dataset_type not in self._cached:
      self._cached[dataset_type] = examples

    pdtb_f.close()
    parse_f.close()

    return examples

def to_level(sense, level=2):
  s_split = sense.split(".")
  s_join = ".".join(s_split[:level])
  return s_join

def get_level(sense):
  s_split = sense.split(".")
  return len(s_split)
