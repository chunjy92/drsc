#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Jayeol Chun'


class PDTBRelation(object):
  def __init__(self, guid, exid, arg1, arg2, label, label_list, conn=None):
    # identification
    self.guid = guid # {dataset_type}-{doc_id}-{unique_id of the relation}
    self.exid = exid # {%d th example in selected dataset}

    # data, list of tokens
    self.arg1 = arg1
    self.arg2 = arg2
    self.conn = conn

    # one-hot label
    self.label = label
    self.label_list = label_list

  def __repr__(self):
    if isinstance(self.arg1, list):

      return f"guid: {self.guid }\tARG1: {' '.join(self.arg1)}" \
        f"\tARg2: {' '.join(self.arg2)}\tConn: {self.conn}" \
        f"\tLabel: {self.label}"
    else:
      return f"guid: {self.guid}\tARG1: {self.arg1}\tARg2: {self.arg2}" \
        f"\tConn: {self.conn}\tLabel: {self.label}"
