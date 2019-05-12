#! /usr/bin/python3
# -*- coding: utf-8 -*-

__author__ = 'Jayeol Chun'


class PDTBRelation(object):
  def __init__(self, guid, exid, arg1, arg2, conn, label, label_list,
               arg1_mask, arg2_mask):
    # identification
    self.guid = guid # {dataset_type}-{unique_id of the relation}
    self.exid = exid # {%d th example in selected dataset}

    # list of tokens (or a string if BERTEmbedding)
    self.arg1 = arg1
    self.arg2 = arg2
    self.conn = conn

    # integer label id
    self.label = label
    self.label_list = label_list

    # attention masks
    self.arg1_attn_mask = arg1_mask
    self.arg2_attn_mask = arg2_mask

  def __repr__(self):
    if isinstance(self.arg1, list):
      return f"guid: {self.guid }\n ARG1: {' '.join(self.arg1)}" \
        f"\n ARG2: {' '.join(self.arg2)}\n Conn: {self.conn}" \
        f"\n Label: {self.label}"
    else:
      return f"guid: {self.guid}\n ARG1: {self.arg1}\n ARG2: {self.arg2}" \
        f"\n Conn: {self.conn}\n Label: {self.label}"
