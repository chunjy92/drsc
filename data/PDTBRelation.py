#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Jayeol Chun'


class PDTBRelation(object):
  def __init__(self, guid, exid, arg1, arg2, label, label_list, conn=None):
    # identification
    self.guid = guid # {dataset_type}-{doc_id}-{unique_id of the relation}
    self.exid = exid # {%d th example in selected dataset}

    # data
    self.arg1 = arg1
    self.arg2 = arg2
    self.conn = conn
    self.label = label
    self.label_list = label_list