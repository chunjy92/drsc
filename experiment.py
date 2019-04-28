#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
import random

import tensorflow as tf

import model
from data import PDTBProcessor
from embedding import BERTEmbedding, Embedding
from utils import const

__author__ = 'Jayeol Chun'


class Experiment(object):
  def __init__(self, hp):
    self.hp = hp

    # init data preprocessor
    self.processor = PDTBProcessor(
      max_arg_length=self.hp.max_arg_length,
      truncation_mode=self.hp.truncation_mode,
      do_lower_case=self.hp.do_lower_case,
      sense_type=self.hp.sense_type,
      sense_level=self.hp.sense_level,
      multiple_senses_action=self.hp.multiple_senses_action
    )

    vocab = None
    if not self.hp.embedding:
      # if no embedding is specified, need to collect vocab from training set
      self.processor.compile_vocab_labels()
      vocab = self.processor.vocab
    else:
      self.processor.compile_labels()

    self.labels = self.processor.labels
    self.label_mapping = self.processor.get_label_mapping()

    # init embedding
    if self.hp.embedding == 'bert':
      bert_model = const.BERT_TEMPLATE.format(self.hp.bert_model)
      bert_config_file = os.path.join(bert_model, const.BERT_CONFIG_FILE)
      bert_init_ckpt = os.path.join(bert_model, const.BERT_CKPT_FILE)
      bert_vocab_file = os.path.join(bert_model, const.BERT_VOCAB_FILE)

      self.embedding = BERTEmbedding(
        model_dir=self.hp.model_dir,
        bert_config_file=bert_config_file,
        vocab_file=bert_vocab_file,
        init_checkpoint=bert_init_ckpt,
        batch_size=self.hp.batch_size,
        max_arg_length=self.hp.max_arg_length,
        truncation_mode=self.hp.truncation_mode,
        do_lower_case=self.hp.do_lower_case,
        use_one_hot_embeddings=self.hp.use_one_hot_embeddings
      )

      # embedding_outputs = self.embedding.run(examples)
      # tf.logging.info("Exporting at " + embedding_pkl_file)
      # with open(embedding_pkl_file, 'wb') as f:
      #   pickle.dump(embedding_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
      if self.hp.embedding == 'googlenews':
        # reduce number of vocab and corresponding vector entries by collecting
        # all unique tokens in dev, test and train
        vocab = self.processor.collect_all_vocab()
        tf.logging.info(f"All Vocab: {len(vocab)}")

      self.embedding = Embedding(embedding=self.hp.embedding,
                                 vocab=vocab,
                                 max_arg_length=self.hp.max_arg_length)

      self.embedding_table = self.embedding.get_embedding_table()
      vocab = self.embedding.get_vocab() # currently not used

    tf.logging.info(f"{self.hp.embedding.upper()} embedding init")

    # init model
    self.build_model()
    tf.logging.info(f"{self.hp.model.upper()} model init")

  def build_model(self):
    if self.hp.model == "mlp":
      self.model = model.MLP(
        labels=self.labels,
        max_arg_length=self.hp.max_arg_length,
        word_vector_width=self.hp.word_vector_width,
        hidden_size=self.hp.hidden_size,
        num_hidden_layers=self.hp.num_hidden_layers,
        learning_rate=self.hp.learning_rate,
        optimizer=self.hp.optimizer,
        sense_type=self.hp.sense_type,
        pooling_action=self.hp.pooling_action,
        embedding_shape=self.embedding_table.shape,
        do_pooling_first=self.hp.do_pooling_first,
        do_finetune_embedding=self.hp.do_finetune_embedding
      )
    else:
      raise NotImplementedError()


  # TODO (April 27)
  def load(self):
    pass

  def run(self):
    if self.hp.do_train:
      tf.logging.info("***** Begin Train *****")
      self.train()
    else:
      self.load()

    self.processor.remove_cache_by_key('train')

    if self.hp.do_eval:
      tf.logging.info("***** Begin Eval *****")
      self.eval()

    self.processor.remove_cache_by_key('dev')

    if self.hp.do_predict:
      tf.logging.info("***** Begin Predict *****")
      self.predict()

  def batchify(self, examples, batch_size, do_shuffle=False):
    if do_shuffle:
      random.shuffle(examples)

    batches = []
    for start in range(0, len(examples), batch_size):
      batches.append(examples[start:start+batch_size])

    return batches

  def train(self):
    examples = self.processor.get_train_examples()
    self.processor.remove_cache_by_key('train')

    init = tf.global_variables_initializer()

    tvars = tf.trainable_variables()
    tf.logging.info("***** Trainable Variables *****")
    for var in tvars:
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

    config = tf.ConfigProto()

    # uncomment to pre-allocate all available GPU memory
    config.gpu_options.allow_growth = True

    # sess handled by Experiment obj, not by each model
    self.sess = tf.Session(config=config)

    # TODO: How to integrate BERT outputs..
    self.sess.run(
      [init, self.model.embedding_init_op],
      feed_dict={self.model.embedding_placeholder: self.embedding_table})

    for iter in range(self.hp.num_iter):
      feat_batches = self.batchify(examples, batch_size=self.hp.batch_size,
                                   do_shuffle=True)
      num_batches = len(feat_batches)
      tf.logging.info(f"Created {num_batches} batches.")

      processed_batches = 0
      for i, batch in enumerate(feat_batches):
        batch = self.embedding.convert_to_ids(batch, self.label_mapping)

        arg1, arg2, conn, label = batch
        _, loss, acc = self.sess.run(
          [self.model.train_op, self.model.loss, self.model.acc],
          feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                     self.model.conn: conn, self.model.label: label})

        processed_batches += 1
        tf.logging.info(
          "[Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
            iter, processed_batches, num_batches, loss, acc))

        if (i+1) % self.hp.eval_every == 0:
          self.eval()

  def eval(self):
    examples = self.processor.get_dev_examples()
    examples = self.embedding.convert_to_ids(examples, self.label_mapping)
    arg1, arg2, conn, label = examples

    loss, acc = self.sess.run(
      [self.model.loss, self.model.acc],
      feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                 self.model.conn: conn, self.model.label: label})

    tf.logging.info("[EVAL] loss: {:.3f} acc: {:.3f}".format(loss, acc))


  def predict(self):
    examples = self.processor.get_test_examples()
    self.processor.remove_cache_by_key('test')

    examples = self.embedding.convert_to_ids(examples, self.label_mapping)
    arg1, arg2, conn, label = examples

    preds, acc = self.sess.run(
      [self.model.preds, self.model.acc],
      feed_dict={self.model.arg1 : arg1, self.model.arg2 : arg2,
                 self.model.conn : conn, self.model.label: label})

    tf.logging.info("[PREDICT] acc: {:.3f}".format(acc))

    # convert label_id preds to labels
    inverse_label_mapping = {v:k for k,v in self.label_mapping.items()}
    preds_str = [inverse_label_mapping[pred] for pred in preds]

    preds_file = os.path.join(self.hp.model_dir, "predictions.txt")

    tf.logging.info(f"Exporting predictions at {preds_file}")
    with open(preds_file, 'w') as f:
      f.write("\n".join(preds_str))
