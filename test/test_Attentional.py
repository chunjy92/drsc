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


class AttentionalModelTest(tf.test.TestCase):
  pass

if __name__ == '__main__':
  tf.test.main()