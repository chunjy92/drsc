#! /usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os

__author__ = 'Jayeol Chun'


def init_logger(folder):
  # get TF logger
  log = logging.getLogger('tensorflow')
  log.setLevel(logging.DEBUG)

  # create formatter and add it to the handlers
  formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

  # create file handler which logs even debug messages
  log_name = "log"
  log_file = os.path.join(folder, log_name+".txt")

  while os.path.exists(log_file):
    if "_" in log_name:
      log_name_split = log_name.split("_")
      i = int(log_name_split[-1])
      log_name = log_name_split[0] + "_{}".format(i + 1)
    else:
      log_name += "_1"
    log_file = os.path.join(folder, log_name+".txt")

  fh = logging.FileHandler(log_file)
  fh.setLevel(logging.DEBUG)
  fh.setFormatter(formatter)
  log.addHandler(fh)
  log.info("Logging at " + log_file)
  return log_file
