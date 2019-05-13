#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.extend([".."])

from data import PDTBProcessor
from data.PDTBProcessor import get_level, to_level

import tensorflow as tf

from utils import const

__author__ = 'Jayeol Chun'


class PDTBProcessorTest(tf.test.TestCase):
  data_dir = os.path.join(os.path.join(os.getcwd(), '..'), const.CONLL)
  processor = PDTBProcessor(
    data_dir=data_dir,
    max_arg_length=64,
    sense_type='implicit',
    sense_level=2,
    multiple_senses_action='pick_first',
    drop_partial_data=False
  )

  dev_examples = processor.get_dev_examples()
  dev_examples_bert = processor.get_dev_examples(for_bert_embedding=True)

  example = dev_examples[9]
  print(example)
  example_bert = dev_examples_bert[9]

  def test_sense_action(self):
    self.assertAllEqual(to_level(self.example.label, 2),
                        to_level(self.example.label_list[0], 2))
    self.assertAllEqual(to_level(self.example_bert.label, 2),
                        to_level(self.example_bert.label_list[0], 2))

  def test_sense_level(self):
    """(May 12) example.label_list does NOT have its senses converted to the
    user-defined sense level value."""
    self.assertAllEqual(get_level(self.example.label), 2)
    # self.assertAllEqual(get_level(self.example.label_list[0]), 2)
    self.assertAllEqual(get_level(self.example_bert.label), 2)
    # self.assertAllEqual(get_level(self.example_bert.label_list[0]), 2)

  def test_sense_labels(self):
    self.assertAllEqual(self.processor.labels, const.IMPLICIT_15_WAY)
    proc = PDTBProcessor(self.data_dir, drop_partial_data=True)
    self.assertAllEqual(proc.labels, const.IMPLICIT_11_WAY)

  def test_tokenization(self):
    gold_arg1_str = "The brokerage firms learned a lesson the last time around"
    gold_arg2_str = "This time , the firms were ready"

    arg1_str = " ".join([x for x in self.example.arg1 if x != const.PAD])
    arg2_str = " ".join([x for x in self.example.arg2 if x != const.PAD])

    self.assertAllEqual(arg1_str, gold_arg1_str)
    self.assertAllEqual(arg2_str, gold_arg2_str)

    self.assertAllEqual(self.example_bert.arg1, gold_arg1_str)
    self.assertAllEqual(self.example_bert.arg2, gold_arg2_str)

  def test_padding(self):
    gold_arg1_str = "The brokerage firms learned a lesson the last time around"
    gold_arg2_str = "This time , the firms were ready"

    gold_arg1_split = gold_arg1_str.split()
    gold_arg2_split = gold_arg2_str.split()

    gold_arg1 = gold_arg1_split + [const.PAD] * (64 - len(gold_arg1_split))
    gold_arg2 = gold_arg2_split + [const.PAD] * (64 - len(gold_arg2_split))

    self.assertAllEqual(self.example.arg1, gold_arg1)
    self.assertAllEqual(self.example.arg2, gold_arg2)

    proc = PDTBProcessor(self.data_dir,
                         max_arg_length=64,
                         padding_action="pad_left_arg1")
    example = proc.get_dev_examples()[9]

    gold_arg1 = [const.PAD] * (64 - len(gold_arg1_split)) + gold_arg1_split
    self.assertAllEqual(example.arg1, gold_arg1)
    self.assertAllEqual(example.arg2, gold_arg2)

  def test_truncation(self):
    proc = PDTBProcessor(self.data_dir,
                         max_arg_length=5,
                         truncation_mode="reverse")
    example = proc.get_dev_examples()[9]

    self.assertAllEqual(" ".join(example.arg1), "lesson the last time around")
    self.assertAllEqual(" ".join(example.arg2), ", the firms were ready")

  def test_mask(self):
    self.assertAllEqual(len(self.example.arg1_attn_mask), 64)
    self.assertAllEqual(len(self.example.arg2_attn_mask), 64)

    gold_arg1_str = "The brokerage firms learned a lesson the last time around"
    gold_arg2_str = "This time , the firms were ready"

    gold_arg1_split = gold_arg1_str.split()
    gold_arg2_split = gold_arg2_str.split()

    arg1_gold_mask = [1] * len(gold_arg1_split) + \
                     [0] * (64 - len(gold_arg1_split))
    arg2_gold_mask = [1] * len(gold_arg2_split) + \
                     [0] * (64 - len(gold_arg2_split))

    self.assertAllEqual(self.example.arg1_attn_mask, arg1_gold_mask)
    self.assertAllEqual(self.example.arg2_attn_mask, arg2_gold_mask)

    proc = PDTBProcessor(self.data_dir,
                         max_arg_length=64,
                         padding_action="pad_left_arg1")
    example = proc.get_dev_examples()[9]
    arg1_gold_mask = [0] * (64 - len(gold_arg1_split)) + \
                     [1] * len(gold_arg1_split)
    self.assertAllEqual(example.arg1_attn_mask, arg1_gold_mask)

if __name__ == '__main__':
  tf.test.main()
