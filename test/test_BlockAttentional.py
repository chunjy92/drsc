#! /usr/bin/python3
# -*- coding: utf-8 -*-
import sys
sys.path.extend([".."])
import random

import numpy as np
import tensorflow as tf

__author__ = 'Jayeol Chun'


class AttentionalModelTest(tf.test.TestCase):
  random.seed(1334)
  np.random.seed(1334)
  tf.random.set_random_seed(1334)

  batch_size = 32
  max_arg_length = 64


  def test_numpy_indexing_reshape(self):
    tgt_batch_size = self.batch_size * 2
    seq_length = self.max_arg_length

    arg1 = np.random.rand(self.batch_size, self.max_arg_length)
    arg2 = np.random.rand(self.batch_size, self.max_arg_length)

    arg = np.zeros([tgt_batch_size, seq_length], dtype=np.float64)
    arg[0::2] = arg1
    arg[1::2] = arg2

    for i,a in enumerate(arg):
      if i % 2 == 0:
        arg1_idx = int(i/2)
        self.assertAllEqual(a, arg1[arg1_idx])
      else:
        arg2_idx = int(i/2)
        self.assertAllEqual(a, arg2[arg2_idx])

    arg_concat = np.reshape(arg, [int(tgt_batch_size/2), int(seq_length*2)])
    arg1_arg2_concat = np.concatenate([arg1, arg2], axis=1)

    self.assertAllEqual(arg_concat, arg1_arg2_concat)

if __name__ == '__main__':
  tf.test.main()