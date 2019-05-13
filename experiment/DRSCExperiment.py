#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
from collections import Counter

import numpy as np
import tensorflow as tf

import embedding
import model
from utils import const
from .Experiment import Experiment

__author__ = 'Jayeol Chun'


class DRSCExperiment(Experiment):
  """
  TODO (May 5): lot of redundancy
  """
  ################################### INIT #####################################
  def init_all(self, is_training=False):
    self.init_embedding(is_training=is_training)
    tf.logging.info("Embedding init")

    self.init_model(is_training=is_training)
    tf.logging.info("Model init")

  def init_embedding(self, is_training=False):
    if self.hp.embedding == 'bert':
      bert_model = const.BERT_TEMPLATE.format(self.hp.bert_model)
      bert_config_file = os.path.join(bert_model, const.BERT_CONFIG_FILE)
      bert_init_ckpt = os.path.join(bert_model, const.BERT_CKPT_FILE)
      bert_vocab_file = os.path.join(bert_model, const.BERT_VOCAB_FILE)

      self.embedding = embedding.BERTEmbedding(
        bert_config_file=bert_config_file,
        vocab_file=bert_vocab_file,
        init_checkpoint=bert_init_ckpt,
        batch_size=self.hp.batch_size,
        max_arg_length=self.hp.max_arg_length,
        truncation_mode=self.hp.truncation_mode,
        do_lower_case=self.hp.do_lower_case,
        finetune_embedding=self.hp.finetune_embedding,
        split_args=self.hp.split_args_in_embedding,
        is_training=is_training,
        padding_action=self.hp.padding_action
      )

      # effectively 768 for base model
      self.hp.word_vector_width = self.embedding.bert_config.hidden_size
      self.embedding_table = None
      self.embedding_shape = None

    else:
      # reduce number of vocab and corresponding vector entries by collecting
      # all unique tokens in dev, test and train, which slows down
      if self.hp.embedding == "random_init":
        # compiles vocab from training set
        self.processor.compile_vocab()
        vocab = self.processor.vocab
      else:
        # compiles vocab from training set AND includes vocab from all other
        # files. We do this because some external embeddings have a large vocab
        # size so words that never appear are filtered out.
        vocab = self.processor.collect_all_vocab(include_blind=True)

      self.embedding = \
        embedding.StaticEmbedding(
          embedding=self.hp.embedding,
          vocab=vocab,
          word_vector_width=self.hp.word_vector_width,
          max_arg_length=self.hp.max_arg_length)

      self.embedding_table = self.embedding.embedding_table
      self.embedding_shape = self.embedding_table.shape

      if self.hp.word_vector_width != self.hp.hidden_size:
        tf.logging.info(
          "[!] `word_vector_width` != `hidden_size`. There will be an "
          "additional linear layer that projects concatenated arg vector (of "
          "length `word_vector_width`) to `hidden_size` dimension using a "
          "learned projection weight matrix.")

  def init_model(self, is_training=False):
    if self.hp.model == "mlp":
      self.model = model.MLP(
        labels=self.labels,
        max_arg_length=self.hp.max_arg_length,
        word_vector_width=self.hp.word_vector_width,
        hidden_size=self.hp.hidden_size,
        num_hidden_layers=self.hp.num_hidden_layers,
        learning_rate=self.hp.learning_rate,
        optimizer=self.hp.optimizer,
        num_train_steps=self.num_train_steps,
        num_warmup_steps=self.num_train_steps,
        sense_type=self.hp.sense_type,
        pooling_action=self.hp.pooling_action,
        conn_action=self.hp.conn_action,
        embedding=self.embedding,
        embedding_name=self.hp.embedding,
        embedding_shape=self.embedding_shape,
        is_training=is_training,
        do_pooling_first=self.hp.do_pooling_first,
        finetune_embedding=self.hp.finetune_embedding,
        split_args_in_embedding=self.hp.split_args_in_embedding
      )
    else:
      self.model = model.Attentional(
        labels=self.labels,
        attention_type=self.hp.model,
        max_arg_length=self.hp.max_arg_length,
        word_vector_width=self.hp.word_vector_width,
        hidden_size=self.hp.hidden_size,
        num_hidden_layers=self.hp.num_hidden_layers,
        num_attention_heads=self.hp.num_attention_heads,
        learning_rate=self.hp.learning_rate,
        optimizer=self.hp.optimizer,
        num_train_steps=self.num_train_steps,
        num_warmup_steps=self.num_train_steps,
        embedding=self.embedding,
        embedding_name=self.hp.embedding,
        embedding_shape=self.embedding_shape,
        is_training=is_training,
        sense_type=self.hp.sense_type,
        pooling_action=self.hp.pooling_action,
        conn_action=self.hp.conn_action,
        do_pooling_first=self.hp.do_pooling_first,
        finetune_embedding=self.hp.finetune_embedding,
        split_args_in_embedding=self.hp.split_args_in_embedding
      )

  ################################### TRAIN ####################################
  def train(self):
    """See docstring for `run`"""

    tf.logging.info("***** Begin Train *****")

    # data
    if self.hp.embedding == 'bert':
      examples = self.processor.get_train_examples(for_bert_embedding=True)
    else:
      examples = self.processor.get_train_examples()

    self.num_train_steps = int(
      len(examples) / self.hp.batch_size * self.hp.num_epochs)
    self.num_warmup_steps = \
      int(self.num_train_steps * self.hp.warmup_proportion)

    global_step = 0
    for epoch in range(self.hp.num_epochs):
      tf.reset_default_graph()
      with tf.Graph().as_default() as graph:
        with tf.Session(config=self.sess_config, graph=graph) as sess:
          model_ckpt = tf.train.latest_checkpoint(self.hp.model_dir)

          self.init_all(is_training=True)
          # for TensorBoard
          self.summary_op = tf.summary.merge_all()

          init = tf.global_variables_initializer()

          saver = None
          if not model_ckpt:
            saver = tf.train.Saver()
            self.init_bert_from_checkpoint()
          else:
            # load from checkpoint
            saver = tf.train.import_meta_graph(self.model_ckpt_path+".meta")
            saver.restore(sess, model_ckpt)

          # run init ops
          sess.run(init)
          if self.hp.embedding != 'bert':
            sess.run(self.model.embedding_init_op,
                     feed_dict={
                       self.model.embedding_placeholder: self.embedding_table})

          feat_batches = self.batchify(examples, self.hp.batch_size, True)
          num_batches = len(feat_batches)
          tf.logging.info(f"Created {num_batches} batches.")

          all_preds = []
          all_correct = []
          all_loss = []

          for i, batch in enumerate(feat_batches):
            batch = self.embedding.convert_to_ids(batch, self.l2i)
            fetch_ops = ['train_op', 'preds', 'per_loss', 'correct']

            ops, feed_dict = self.model.postprocess_batch(
              batch, fetch_ops=fetch_ops)
            _, preds, per_loss, correct = sess.run(
              ops, feed_dict=feed_dict)

            # self.summary_writer.add_summary(summary, global_step)
            global_step += 1

            all_preds.extend(preds)
            all_loss.extend(per_loss)
            all_correct.extend(correct)

            if (i+1) % self.hp.log_every == 0:
              c = Counter(all_preds)

              mean_loss = np.mean(all_loss)
              mean_acc = np.mean(all_correct)

              msg_header = \
                "[Epoch {} Batch {}/{}]".format(epoch, i+1, num_batches)
              self.display_log(c, msg_header, mean_loss, mean_acc)

              all_preds = []
              all_correct = []
              all_loss = []

          tf.logging.info("Saving model parameters")
          saver.save(sess, self.model_ckpt_path)

      self.eval()

  ################################### EVAL #####################################
  def eval(self):
    """See docstring for `run`"""

    tf.logging.info("***** Begin Eval *****")

    if self.hp.embedding == "bert":
      examples = self.processor.get_dev_examples(for_bert_embedding=True)
    else:
      examples = self.processor.get_dev_examples()

    # build graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
      with tf.Session(config=self.sess_config, graph=graph) as sess:
        self.init_all(is_training=False)

        saver = tf.train.Saver()
        saver.restore(sess, self.model_ckpt_path)

        example_batches = self.batchify(examples, self.hp.batch_size, False)
        num_batches = len(example_batches)
        tf.logging.info(f"Created {num_batches} batches.")

        all_correct = []
        all_loss = []
        all_counter = Counter()

        for i, batch in enumerate(example_batches):
          batch = self.embedding.convert_to_ids(batch, self.l2i)
          fetch_ops = ['preds', 'per_loss', 'mean_loss', 'correct', 'acc']
          ops, feed_dict = self.model.postprocess_batch(batch,
                                                        fetch_ops=fetch_ops)
          preds, per_example_loss, loss, correct, acc = \
            sess.run(ops, feed_dict=feed_dict)

          # accumulate losses and prediction evaluation
          all_loss.extend(per_example_loss)
          all_correct.extend(correct)

          c = Counter(preds)
          all_counter.update(c)

          msg_header = "[EVAL Batch {}/{}]".format(i+1, num_batches)
          self.display_log(c, msg_header, loss, acc)

        msg_header = "[EVAL FINAL]"
        mean_loss = np.mean(all_loss)
        mean_acc = np.mean(all_correct)
        self.display_log(all_counter, msg_header, mean_loss, mean_acc)

  ################################## PREDICT ###################################
  def predict(self):
    """See docstring for `run`

    TODO: As of May 5, no longer exporting predictions.

    Returns:
      Dict of {str of dataset_type, list of model_predictions} key-value pair
    """

    tf.logging.info("***** Begin Predict *****")

    res = {}

    # build graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
      with tf.Session(config=self.sess_config, graph=graph) as sess:
        self.init_all(is_training=False)

        saver = tf.train.Saver()
        saver.restore(sess, self.model_ckpt_path)

        for dataset_type in ['test', 'blind']:
          is_test_set = dataset_type == 'test'

          if self.hp.embedding == 'bert':
            if is_test_set:
              examples = \
                self.processor.get_test_examples(for_bert_embedding=True)
            else:
              examples = self.processor.get_blind_examples(
                for_bert_embedding=True)
          else:
            if is_test_set:
              examples = self.processor.get_test_examples()
            else:
              examples = self.processor.get_blind_examples()

          tf.logging.info(f"Inference on {dataset_type.upper()} set")

          example_batches = self.batchify(examples, self.hp.batch_size, False)
          num_batches = len(example_batches)
          tf.logging.info(f"Created {num_batches} batches.")

          all_correct = []
          all_preds = []
          all_counter = Counter()

          for i, batch in enumerate(example_batches):
            batch = self.embedding.convert_to_ids(batch, self.l2i)
            fetch_ops = ['preds', 'correct', 'acc']
            ops, feed_dict = self.model.postprocess_batch(batch,
                                                          fetch_ops=fetch_ops)
            preds, correct, acc = sess.run(ops, feed_dict=feed_dict)

            all_preds.extend(preds)
            all_correct.extend(correct)

            c = Counter(preds)
            all_counter.update(c)

            msg_header = \
              "[{} Batch {}/{}]".format(dataset_type.upper(), i+1, num_batches)
            self.display_log(c, msg_header, mean_loss=None, mean_acc=acc)

          msg_header = f"[{dataset_type.upper()} FINAL]"
          mean_acc = np.mean(all_correct)
          self.display_log(all_counter, msg_header, mean_loss=None,
                           mean_acc=mean_acc)

          # convert label_id preds to labels
          preds_str = [self.i2l(pred) for pred in all_preds]
          res[dataset_type] = preds_str

    return res



