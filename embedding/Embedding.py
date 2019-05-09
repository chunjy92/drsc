#! /usr/bin/python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

__author__ = 'Jayeol Chun'


class Embedding(ABC):
  ############################ Absract methods #################################
  @abstractmethod
  def convert_to_ids(self, examples, l2i, **kwargs):
    pass

  @abstractmethod
  def convert_to_values(self, examples, **kwargs):
    pass

  ################################## GETTERS ###################################
  @property
  def embedding_table(self):
    return self._embedding_table

  @property
  def vocab(self):
    return self._vocab
