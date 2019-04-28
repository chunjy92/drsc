#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

__author__ = 'Jayeol Chun'


class HParams(object):
  """Thin wrapper over tf HParams

  in case tf HParams becomes necessary later.
  """
  def __init__(self, flags):
    self._params = tf.contrib.training.HParams()

    for k,v in vars(flags).items():
      self._params.add_hparam(k,v)
      self.__setattr__(k, v)

  def __getitem__(self, item):
    return self._params.get(item)

  @property
  def get_params(self):
    return self._params
