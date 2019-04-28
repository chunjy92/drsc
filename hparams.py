#! /usr/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf

__author__ = 'Jayeol Chun'


class HParams(object):
  """Thin wrapper over tf HParams

  in case tf HParams becomes necessary later.
  """
  def __init__(self, flags):
    self._params = tf.contrib.training.HParams(
      # paths
      model_dir=flags.model_dir,

      # embedding specification
      embedding=flags.embedding,

      # model specification
      model=flags.model

    )

    # make attr
    for k,v in self._params.values().items():
      self.__setattr__(k, v)

  def __getitem__(self, item):
    return self._params.get(item)

  @property
  def get_params(self):
    return self._params
