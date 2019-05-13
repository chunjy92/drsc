#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.extend([".."])

import tensorflow as tf

from bert import tokenization
from data import PDTBProcessor
from embedding import BERTEmbedding
from utils import const

__author__ = 'Jayeol Chun'


class BERTTEmbeddingTest(tf.test.TestCase):
  data_dir = os.path.join(os.path.join(os.getcwd(), '..'), const.CONLL)
  processor = PDTBProcessor(
    data_dir=data_dir,
    max_arg_length=64,
    sense_type='implicit',
    sense_level=2,
    multiple_senses_action='pick_first',
    drop_partial_data=False
  )

  labels = processor.labels

  example = processor.get_dev_examples(for_bert_embedding=True)[9]
  print(example)

  bert_model = const.BERT_TEMPLATE.format('uncased')
  bert_config_file = os.path.join(bert_model, const.BERT_CONFIG_FILE)
  bert_init_ckpt = os.path.join(bert_model, const.BERT_CKPT_FILE)
  bert_vocab_file = os.path.join(bert_model, const.BERT_VOCAB_FILE)

  embedding = BERTEmbedding(
    bert_config_file=bert_config_file,
    vocab_file=bert_vocab_file,
    init_checkpoint=bert_init_ckpt,
    max_arg_length=64,
    do_lower_case=True
  )

  bert_tokenizer = tokenization.FullTokenizer(vocab_file=bert_vocab_file,
                                              do_lower_case=True)

  def test_tokenizer(self):
    arg1_toks = self.bert_tokenizer.tokenize(self.example.arg1)
    arg2_toks = self.bert_tokenizer.tokenize(self.example.arg2)

    gold_arg1 = ['the', 'broker', '##age', 'firms', 'learned', 'a', 'lesson',
                 'the', 'last', 'time', 'around']
    gold_arg2 = ['this', 'time', ',', 'the', 'firms', 'were', 'ready']

    self.assertAllEqual(arg1_toks, gold_arg1)
    self.assertAllEqual(arg2_toks, gold_arg2)

  def test_convert_to_ids(self):
    # label to index
    l2i = lambda l: self.labels.index(l)

    example_ids = self.embedding.convert_to_ids([self.example], l2i)
    arg1, arg2, conn, label, arg1_mask, arg2_mask = example_ids

    arg1_toks = self.bert_tokenizer.tokenize(self.example.arg1)
    arg2_toks = self.bert_tokenizer.tokenize(self.example.arg2)

    # [CLS] ~~~ [SEP]
    arg1_bert = ['[CLS]'] + arg1_toks + ['[SEP]']
    arg2_bert = ['[CLS]'] + arg2_toks + ['[SEP]']

    # real token ids
    arg1_bert_ids = self.bert_tokenizer.convert_tokens_to_ids(arg1_bert)
    arg2_bert_ids = self.bert_tokenizer.convert_tokens_to_ids(arg2_bert)

    # padding token ids: 0
    arg1_bert_ids_padding = arg1_bert_ids + [0] * (64 - len(arg1_bert_ids))
    arg2_bert_ids_padding = arg2_bert_ids + [0] * (64 - len(arg2_bert_ids))

    self.assertAllEqual(arg1[0], arg1_bert_ids_padding)
    self.assertAllEqual(arg2[0], arg2_bert_ids_padding)

    gold_arg1_mask = [1] * len(arg1_bert_ids) + [0] * (64 - len(arg1_bert_ids))
    gold_arg2_mask = [1] * len(arg2_bert_ids) + [0] * (64 - len(arg2_bert_ids))

    self.assertAllEqual(arg1_mask[0], gold_arg1_mask)
    self.assertAllEqual(arg2_mask[0], gold_arg2_mask)

    self.assertAllEqual(l2i(self.example.label), label[0])

  def test_padding_truncation(self):
    # label to index
    l2i = lambda l: self.labels.index(l)

    # padding
    embedding = BERTEmbedding(
      bert_config_file=self.bert_config_file,
      vocab_file=self.bert_vocab_file,
      init_checkpoint=self.bert_init_ckpt,
      max_arg_length=11,
      do_lower_case=True,
      padding_action="pad_left_arg1",
      truncation_mode='reverse'
    )

    example_ids = embedding.convert_to_ids([self.example], l2i)
    arg1, arg2, _, _, _, _ = example_ids

    arg1_toks = self.bert_tokenizer.tokenize(self.example.arg1)[-9:]
    arg2_toks = self.bert_tokenizer.tokenize(self.example.arg2)[-9:]

    # [CLS] ~~~ [SEP]
    arg1_bert = ['[CLS]'] + arg1_toks + ['[SEP]']
    arg2_bert = ['[CLS]'] + arg2_toks + ['[SEP]']

    # real token ids
    arg1_bert_ids = self.bert_tokenizer.convert_tokens_to_ids(arg1_bert)
    arg2_bert_ids = self.bert_tokenizer.convert_tokens_to_ids(arg2_bert)

    # padding token ids: 0
    arg1_bert_ids_padding = [arg1_bert_ids[0]] + \
                            [0] * (11 - len(arg1_bert_ids)) + \
                            arg1_bert_ids[1:]
    arg2_bert_ids_padding = arg2_bert_ids + [0] * (11 - len(arg2_bert_ids))

    self.assertAllEqual(arg1[0], arg1_bert_ids_padding)
    self.assertAllEqual(arg2[0], arg2_bert_ids_padding)

if __name__ == '__main__':
  tf.test.main()
