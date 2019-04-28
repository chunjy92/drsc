#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Jayeol Chun'


class BERTInputFeatures(object):
  pass


class InputFeatures(object):
  def __init__(self, arg1, arg2, conn, label):
    self.arg1 = arg1
    self.arg2 = arg2
    self.conn = conn
    self.label = label
