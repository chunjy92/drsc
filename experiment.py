#! /usr/bin/python3
# -*- coding: utf-8 -*-
from data import PDTBProcessor

__author__ = 'Jayeol Chun'


class Experiment(object):
  def __init__(self, hp):
    self.hp = hp

    self.processor = PDTBProcessor()

    # init data preprocessor
    if not self.hp.embedding:
      self.processor.compile_vocab_labels()
    else:
      self.processor.compile_labels()

    print(len(self.processor.vocab))
    print(len(self.processor.labels))
    print(self.processor.labels)

    # init embedding

    # build model

  def run(self):
    pass
