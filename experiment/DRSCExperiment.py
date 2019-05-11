#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
from collections import Counter

import numpy as np
import tensorflow as tf

import embedding
import model
from bert import modeling
from utils import const
from .Experiment import Experiment

__author__ = 'Jayeol Chun'


class DRSCExperiment(Experiment):
  """
  TODO (May 5): lot of redundancy
  """
  ################################### INIT #####################################
  def init_all(self, is_training=False):
    if not is_training:
      self.num_train_steps = 0
      self.num_warmup_steps = 0

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
        model_dir=self.hp.model_dir,
        bert_config_file=bert_config_file,
        vocab_file=bert_vocab_file,
        init_checkpoint=bert_init_ckpt,
        batch_size=self.hp.batch_size,
        max_arg_length=self.hp.max_arg_length,
        truncation_mode=self.hp.truncation_mode,
        do_lower_case=self.hp.do_lower_case,
        finetune_embedding=self.hp.finetune_embedding,
        is_training=is_training,
        padding_action=self.hp.padding_action,
        use_one_hot_embeddings=self.hp.use_one_hot_embeddings
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
          embedding=self.hp.embedding, vocab=vocab,
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

    tf.logging.info(f"Vocab size: {len(self.embedding.vocab)}")

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
        finetune_embedding=self.hp.finetune_embedding
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
        finetune_embedding=self.hp.finetune_embedding
      )

  ################################### TRAIN ####################################
  def train(self):
    """See docstring for `run`"""

    tf.logging.info("***** Begin Train *****")

    # data
    train_fn = None
    if self.hp.embedding == 'bert' and not self.hp.finetune_embedding:
      examples = self.processor.get_train_examples(for_bert_embedding=True)
      train_fn = self.train_from_vals
    else:
      if self.hp.embedding == 'bert':
        examples = self.processor.get_train_examples(for_bert_embedding=True)
      else:
        examples = self.processor.get_train_examples()
      train_fn = self.train_from_ids

    self.processor.remove_cache_by_key('train')

    self.num_train_steps = int(
      len(examples) / self.hp.batch_size * self.hp.num_epochs)
    self.num_warmup_steps = \
      int(self.num_train_steps * self.hp.warmup_proportion)

    train_fn(examples)

  def train_from_ids(self, examples):
    global_step = 0
    for epoch in range(self.hp.num_epochs):

      tf.reset_default_graph()
      with tf.Graph().as_default() as graph:
        with tf.Session(config=self.sess_config, graph=graph) as sess:
          model_ckpt = tf.train.latest_checkpoint(self.hp.model_dir)

          self.init_all(is_training=True)
          init = tf.global_variables_initializer()

          if not model_ckpt:
            # for TensorBoard
            self.summary_op = tf.summary.merge_all()

            saver = tf.train.Saver()

            tvars = tf.trainable_variables()

            if self.hp.embedding == "bert" and self.hp.finetune_embedding:
              initialized_variable_names = []
              (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(
                  tvars, self.embedding.init_checkpoint)

              tf.train.init_from_checkpoint(
                self.embedding.init_checkpoint, assignment_map)

              tf.logging.info("**** Trainable Variables ****")
              for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                  init_string = ", *INIT_FROM_CKPT*"
                tf.logging.info("  name = %s, shape = %s%s", var.name,
                                var.shape, init_string)

          else:
            # load from checkpoint
            saver = tf.train.import_meta_graph(self.model_ckpt_path+".meta")
            saver.restore(sess, model_ckpt)

          # run init ops
          sess.run(init)
          if self.hp.embedding != 'bert':
            # no embedding lookup necessary for bert embedding
            sess.run(self.model.embedding_init_op,
                     feed_dict={
                       self.model.embedding_placeholder: self.embedding_table})

          feat_batches = self.batchify(examples,
                                       batch_size=self.hp.batch_size,
                                       do_shuffle=True)
          num_batches = len(feat_batches)
          tf.logging.info(f"Created {num_batches} batches.")

          for i, batch in enumerate(feat_batches):
            batch = self.embedding.convert_to_ids(batch, self.l2i)
            fetch_ops = ['train_op', 'preds', 'mean_loss', 'acc']

            ops, feed_dict = \
              self.model.postprocess_batch_ids(batch, fetch_ops=fetch_ops)

            _, preds, loss, acc = sess.run(ops, feed_dict=feed_dict)
            # self.summary_writer.add_summary(summary, global_step)
            global_step += 1

            if (i+1) % self.hp.log_every == 0:
              c = Counter(preds)
              for key in c.keys():
                tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

              tf.logging.info(
                "[Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
                  epoch, i+1, num_batches, loss, acc))

          # save
          saver.save(sess, self.model_ckpt_path)

      self.eval()


  def train_from_vals(self, examples):

    embedding_res = self.embedding.convert_to_values(examples)
    bert_outputs, exid_to_feature_mapping = embedding_res

    global_step = 0
    for epoch in range(self.hp.num_epochs):
      bert_batches = self.batchify(examples,
                                   batch_size=self.hp.batch_size,
                                   do_shuffle=True)
      num_batches = len(bert_batches)
      tf.logging.info(f"Created {num_batches} batches.")

      for i, batch in enumerate(bert_batches):
        feed_dict = self.model.postprocess_batch_vals(
          batch, values=bert_outputs, l2i_mapping=self.l2i,
          exid_to_feature_mapping=exid_to_feature_mapping)

        _, preds, loss, acc, summary = sess.run(
          [self.model.train_op, self.model.preds, self.model.loss,
           self.model.acc, self.summary_op], feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step)
        global_step += 1

        if (i+1) % self.hp.log_every == 0:
          c = Counter(preds)
          for key in c.keys():
            tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

          tf.logging.info(
            "[TRAIN Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
              epoch, i+1, num_batches, loss, acc))

  ################################### EVAL #####################################
  def eval(self):
    """See docstring for `run`"""

    tf.logging.info("***** Begin Eval *****")

    eval_fn = None
    if self.hp.embedding == 'bert' and not self.hp.finetune_embedding:
      examples = self.processor.get_dev_examples(for_bert_embedding=True)
      eval_fn = self.eval_from_vals
    else:
      if self.hp.embedding == "bert":
        examples = self.processor.get_dev_examples(for_bert_embedding=True)
      else:
        examples = self.processor.get_dev_examples()
      eval_fn = self.eval_from_ids

    # build graph
    tf.reset_default_graph()
    with tf.Graph().as_default() as graph:
      with tf.Session(config=self.sess_config, graph=graph) as sess:
        self.init_all(is_training=False)

        saver = tf.train.Saver()

        saver.restore(sess, self.model_ckpt_path)
        eval_fn(examples, sess)

  def eval_from_ids(self, examples, sess):

    example_batches = self.batchify(examples,
                                    batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)
    tf.logging.info(f"Created {num_batches} batches.")

    all_correct = []
    all_loss = []
    all_counter = Counter()

    for i, batch in enumerate(example_batches):
      batch = self.embedding.convert_to_ids(batch, self.l2i)
      fetch_ops = ['preds', 'per_loss', 'mean_loss', 'correct', 'acc']
      ops, feed_dict = self.model.postprocess_batch_ids(batch,
                                                        fetch_ops=fetch_ops)

      preds, per_example_loss, loss, correct, acc = \
        sess.run(ops, feed_dict=feed_dict)

      # accumulate losses and prediction evaluation
      all_loss.extend(per_example_loss)
      all_correct.extend(correct)

      c = Counter(preds)
      all_counter.update(c)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

      tf.logging.info(
        "[EVAL Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
          i + 1, num_batches, loss, acc))

    tf.logging.info("[EVAL FINAL] loss: {:.3f} acc: {:.3f}".format(
      np.mean(all_loss), np.mean(all_correct)))

    # all pred outputs for dev set
    for key in all_counter.keys():
      tf.logging.info(" {:3d}: {}".format(all_counter[key], self.labels[key]))

  def eval_from_vals(self, examples, sess):
    embedding_res = self.embedding.convert_to_values(examples)
    bert_outputs, exid_to_feature_mapping = embedding_res

    example_batches = self.batchify(examples,
                                    batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)
    tf.logging.info(f"Created {num_batches} batches.")

    # collect each prediction output result, where 0: incorrect and 1: correct
    # also collect loss associated with each prediction
    # * THIS IS NECESSARY only because bert outputs for the entire batch of
    #  ~1000 relations will cause OOM issue on GPU, so move to batchified
    #  setting and handle outputs accordingly.
    all_correct = []
    all_loss = []
    all_counter = Counter()
    for i, batch in enumerate(example_batches):
      feed_dict = self.model.postprocess_batch_vals(
        batch, values=bert_outputs, l2i_mapping=self.l2i,
        exid_to_feature_mapping=exid_to_feature_mapping)

      per_example_loss, loss, preds, correct, acc = \
        sess.run(
          [self.model.per_example_loss, self.model.loss, self.model.preds,
           self.model.correct, self.model.acc], feed_dict=feed_dict)

      # accumulate losses and prediction evaluation
      all_loss.extend(per_example_loss)
      all_correct.extend(correct)

      c = Counter(preds)
      all_counter.update(c)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

      tf.logging.info(
        "[EVAL Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
          i + 1, num_batches, loss, acc))

    tf.logging.info("[EVAL FINAL] loss: {:.3f} acc: {:.3f}".format(
      np.mean(all_loss), np.mean(all_correct)))

    # all pred outputs for dev set
    for key in all_counter.keys():
      tf.logging.info(" {:3d}: {}".format(all_counter[key], self.labels[key]))

  ################################## PREDICT ###################################
  def predict(self):
    """See docstring for `run`

    TODO: As of May 5, no longer exporting predictions.

    Returns:
      Dict of {str of dataset_type, list of model_predictions} key-value pair
    """

    tf.logging.info("***** Begin Predict *****")

    # # build graph
    # tf.reset_default_graph()
    # self.init_all(is_training=False)
    #
    # saver = tf.train.Saver()
    #
    # sess = tf.Session(config=self.sess_config)
    # saver.restore(sess, self.model_ckpt_path)
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

          predict_fn = None
          if self.hp.embedding == 'bert' and not self.hp.finetune_embedding:
            if is_test_set:
              examples = self.processor.get_test_examples(for_bert_embedding=True)
            else:
              examples = self.processor.get_blind_examples(for_bert_embedding=True)
            predict_fn = self.predict_from_vals
          else:
            if self.hp.embedding == 'bert':
              if is_test_set:
                examples = self.processor.get_test_examples(for_bert_embedding=True)
              else:
                examples = self.processor.get_blind_examples(
                  for_bert_embedding=True)
            else:
              if is_test_set:
                examples = self.processor.get_test_examples()
              else:
                examples = self.processor.get_blind_examples()
            predict_fn = self.predict_from_ids

          tf.logging.info(f"Inference on {dataset_type.upper()} set")
          self.processor.remove_cache_by_key(dataset_type)

          preds = predict_fn(examples, sess, is_test_set=is_test_set)
          res[dataset_type] = preds

    return res

  def predict_from_ids(self, examples, sess, is_test_set=False):
    if is_test_set:
      dataset_type = "test"
    else:
      dataset_type = "blind"

    example_batches = self.batchify(examples,
                                    batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)
    tf.logging.info(f"Created {num_batches} batches.")

    all_correct = []
    all_preds = []
    all_counter = Counter()

    for i, batch in enumerate(example_batches):
      batch = self.embedding.convert_to_ids(batch, self.l2i)
      fetch_ops = ['preds', 'correct', 'acc']
      ops, feed_dict = self.model.postprocess_batch_ids(batch,
                                                        fetch_ops=fetch_ops)

      preds, correct, acc = sess.run(ops, feed_dict=feed_dict)

      all_preds.extend(preds)
      all_correct.extend(correct)

      c = Counter(preds)
      all_counter.update(c)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

      tf.logging.info(
        "[{} Batch {}/{}] acc: {:.3f}".format(
          dataset_type.upper(), i + 1, num_batches, acc))

    tf.logging.info("[{} FINAL] acc: {:.3f}".format(
      dataset_type.upper(), np.mean(all_correct)))

    for key in all_counter.keys():
      tf.logging.info(f" {self.labels[key]}: {all_counter[key]}")

    # convert label_id preds to labels
    preds_str = [self.i2l(pred) for pred in all_preds]
    return preds_str

  def predict_from_vals(self, examples, sess, is_test_set=False):
    if is_test_set:
      dataset_type = "test"
    else:
      dataset_type = "blind"

    embedding_res = self.embedding.convert_to_values(examples)
    bert_outputs, exid_to_feature_mapping = embedding_res

    example_batches = self.batchify(examples,
                                    batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)
    tf.logging.info(f"Created {num_batches} batches.")

    all_correct = []
    all_preds = []
    all_counter = Counter()

    for i, batch in enumerate(example_batches):
      feed_dict = self.model.postprocess_batch_vals(
        batch, values=bert_outputs, l2i_mapping=self.l2i,
        exid_to_feature_mapping=exid_to_feature_mapping)

      preds, correct, acc = \
        sess.run([self.model.preds, self.model.correct, self.model.acc],
                 feed_dict=feed_dict)

      all_preds.extend(preds)
      all_correct.extend(correct)

      c = Counter(preds)
      all_counter.update(c)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

      tf.logging.info(
        "[{} Batch {}/{}] acc: {:.3f}".format(
          dataset_type.upper(), i + 1, num_batches,  acc))

    tf.logging.info("[{} FINAL] acc: {:.3f}".format(
      dataset_type.upper(), np.mean(all_correct)))

    for key in all_counter.keys():
      tf.logging.info(f" {self.labels[key]}: {all_counter[key]}")

    # convert preds to labels
    preds_str = [self.i2l(pred) for pred in all_preds]

    return preds_str
