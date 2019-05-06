#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
from collections import Counter

import numpy as np
import tensorflow as tf

import model
from embedding import BERTEmbedding, Embedding
from utils import const
from .Experiment import Experiment

__author__ = 'Jayeol Chun'


class DRSCExperiment(Experiment):
  ################################### INIT #####################################
  def init_embedding(self):
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

      self.hp.word_vector_width = 768
      self.embedding_table = None
      self.embedding_shape = None

    else:
      # reduce number of vocab and corresponding vector entries by collecting
      # all unique tokens in dev, test and train, which slows down
      vocab = self.processor.collect_all_vocab()
      tf.logging.debug(f"All Vocab: {len(vocab)}")

      self.embedding = Embedding(embedding=self.hp.embedding,
                                 vocab=vocab,
                                 word_vector_width=self.hp.word_vector_width,
                                 max_arg_length=self.hp.max_arg_length)

      self.embedding_table = self.embedding.get_embedding_table()
      self.embedding_shape = self.embedding_table.shape

      # if self.hp.word_vector_width != self.hp.hidden_size:
      #   # hidden_size
      #   pass

  def init_model(self):
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
        embedding=self.hp.embedding,
        embedding_shape=self.embedding_shape,
        do_pooling_first=self.hp.do_pooling_first,
        do_finetune_embedding=self.hp.do_finetune_embedding
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
        embedding=self.hp.embedding,
        embedding_shape=self.embedding_shape,
        optimizer=self.hp.optimizer,
        sense_type=self.hp.sense_type,
        pooling_action=self.hp.pooling_action,
        do_pooling_first=self.hp.do_pooling_first,
        do_finetune_embedding=self.hp.do_finetune_embedding
      )

  ################################### TRAIN ####################################
  def train(self):
    """See docstring for `run`"""

    train_fn = None
    if self.hp.embedding == 'bert':
      examples = self.processor.get_train_examples(for_bert_embedding=True)
      train_fn = self.train_from_vals
    else:
      examples = self.processor.get_train_examples()
      train_fn = self.train_from_ids

    self.processor.remove_cache_by_key('train')

    self.init = tf.global_variables_initializer()

    tvars = tf.trainable_variables()
    tf.logging.info("***** Trainable Variables *****")
    for var in tvars:
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

    config = tf.ConfigProto()

    # if False,pre-allocates all available GPU memory
    config.gpu_options.allow_growth = self.hp.allow_gpu_growth

    # sess handled by Experiment obj, not by each model
    self.sess = tf.Session(config=config)

    train_fn(examples)

  def train_from_ids(self, examples):
    self.sess.run(
      [self.init, self.model.embedding_init_op],
      feed_dict={self.model.embedding_placeholder: self.embedding_table})

    for iter in range(self.hp.num_epochs):
      feat_batches = self.batchify(examples, batch_size=self.hp.batch_size,
                                   do_shuffle=True)
      num_batches = len(feat_batches)
      tf.logging.info(f"Created {num_batches} batches.")

      processed_batches = 0
      for i, batch in enumerate(feat_batches):
        batch_ids = self.embedding.convert_to_ids(batch, self.l2i)
        arg1, arg2, conn, label_ids = batch_ids

        arg1_mask = []
        arg2_mask = []
        for batch_example in batch:
          batch_example_arg1 = batch_example.arg1
          batch_example_arg1_mask = []
          for arg1_token in batch_example_arg1:
            if arg1_token == const.PAD:
              batch_example_arg1_mask.append(0)
            else:
              batch_example_arg1_mask.append(1)
          arg1_mask.append(batch_example_arg1_mask)

          batch_example_arg2 = batch_example.arg2
          batch_example_arg2_mask = []
          for arg2_token in batch_example_arg2:
            if arg2_token == const.PAD:
              batch_example_arg2_mask.append(0)
            else:
              batch_example_arg2_mask.append(1)
          arg2_mask.append(batch_example_arg2_mask)

        _, preds, loss, acc = self.sess.run(
          [self.model.train_op, self.model.preds, self.model.loss, self.model.acc],
          feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                     self.model.conn: conn, self.model.label: label_ids,
                     self.model.arg1_attn_mask: arg1_mask,
                     self.model.arg2_attn_mask: arg2_mask})
        c = Counter(preds)
        for key in c.keys():
          tf.logging.info(self.labels[key])
        processed_batches += 1
        tf.logging.info(
          "[Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f} {}".format(
            iter, processed_batches, num_batches, loss, acc, c))

        if self.hp.eval_every > 0 and (i + 1) % self.hp.eval_every == 0:
          self.eval()

      if self.hp.eval_every < 0:
        # eval every end of iteration
        self.eval()

  def train_from_vals(self, examples):
    bert_outputs = self.embedding.run(examples)

    # if isinstance(self.embedding, BERTEmbedding):
    bert_outputs, exid_to_feature_mapping = bert_outputs

    # coupling of these in prep for random shuffling
    bert_examples_pair = \
      [(example, bert_output)
       for example, bert_output in zip(examples, bert_outputs)]

    self.sess.run(self.init)

    for iter in range(self.hp.num_epochs):
      bert_batches = self.batchify(bert_examples_pair,
                                   batch_size=self.hp.batch_size,
                                   do_shuffle=True)
      num_batches = len(bert_batches)

      for i, batch in enumerate(bert_batches):
        # tedious decoupling
        label_ids = []
        batch_bert_outputs = []
        arg1_mask = []
        arg2_mask = []
        for batch_example, batch_bert_output in batch:
          batch_bert_outputs.append(batch_bert_output)

          # label
          label_ids.append(self.l2i(batch_example.label))

          # mask
          batch_example_id = batch_example.exid

          batch_arg1_feature = exid_to_feature_mapping[batch_example_id][0]
          batch_arg1_mask = batch_arg1_feature.input_mask
          arg1_mask.append(batch_arg1_mask)

          batch_arg2_feature = exid_to_feature_mapping[batch_example_id][1]
          batch_arg2_mask = batch_arg2_feature.input_mask
          arg2_mask.append(batch_arg2_mask)

        # prepare bert output: [batch, total_seq_length, bert_hidden_size]
        batch_bert_outputs = np.asarray(batch_bert_outputs)
        total_seq_length = batch_bert_outputs.shape[1]
        assert total_seq_length == self.hp.max_arg_length * 2, \
          "Sequence length mismatch between BERT output and parameter"

        arg1 = batch_bert_outputs[:, :self.hp.max_arg_length, :]
        arg2 = batch_bert_outputs[:, self.hp.max_arg_length:, :]

        # TODO: connectives
        # since we don't care about connectives, make them 0 for now
        conn = np.zeros([self.hp.batch_size,
                         self.hp.max_arg_length,
                         batch_bert_outputs.shape[-1]])

        _, preds, loss, acc = self.sess.run(
          [self.model.train_op, self.model.preds, self.model.loss,
           self.model.acc],
          feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                     self.model.conn: conn, self.model.label: label_ids,
                     self.model.arg1_attn_mask: arg1_mask,
                     self.model.arg2_attn_mask: arg2_mask})
        c = Counter(preds)
        for key in c.keys():
          tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

        tf.logging.info(
          "[TRAIN Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
            iter, i+1, num_batches, loss, acc))

        if self.hp.eval_every > 0 and (i + 1) % self.hp.eval_every == 0:
          self.eval()

      if self.hp.eval_every < 0:
        # eval every end of iteration
        self.eval()

  ################################### EVAL #####################################
  def eval(self):
    """See docstring for `run`"""

    eval_fn = None
    if self.hp.embedding == 'bert':
      examples = self.processor.get_dev_examples(for_bert_embedding=True)
      eval_fn = self.eval_from_vals
    else:
      examples = self.processor.get_dev_examples()
      eval_fn = self.eval_from_ids

    eval_fn(examples)

  def eval_from_ids(self, examples):

    example_ids = self.embedding.convert_to_ids(examples, self.l2i)
    arg1, arg2, conn, label = example_ids

    arg1_mask = []
    arg2_mask = []
    for example in examples:
      example_arg1 = example.arg1
      example_arg1_mask = []
      for arg1_token in example_arg1:
        if arg1_token == const.PAD:
          example_arg1_mask.append(0)
        else:
          example_arg1_mask.append(1)
      arg1_mask.append(example_arg1_mask)

      example_arg2 = example.arg2
      example_arg2_mask = []
      for arg2_token in example_arg2:
        if arg2_token == const.PAD:
          example_arg2_mask.append(0)
        else:
          example_arg2_mask.append(1)
      arg2_mask.append(example_arg2_mask)

    loss, preds, acc = self.sess.run(
      [self.model.loss, self.model.preds, self.model.acc],
      feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                 self.model.conn: conn, self.model.label: label,
                 self.model.arg1_attn_mask: arg1_mask,
                 self.model.arg2_attn_mask: arg2_mask})

    tf.logging.info("[EVAL] loss: {:.3f} acc: {:.3f}".format(loss, acc))
    c = Counter(preds)
    for key in c.keys():
      tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

  def eval_from_vals(self, examples):
    bert_outputs = self.embedding.run(examples)

    bert_outputs, exid_to_feature_mapping = bert_outputs

    example_batches = self.batchify(examples, batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)

    all_correct = []
    all_loss = []
    all_counter = Counter()
    for i, example_batch in enumerate(example_batches):
      label_ids = []
      bert_output_batch = []
      arg1_mask, arg2_mask = [], []

      for example in example_batch:
        # arg1 and arg2
        bert_output_batch.append(bert_outputs[example.exid])

        label_ids.append(self.l2i(example.label))

        exid = example.exid
        dev_arg1_feature = exid_to_feature_mapping[exid][0]
        dev_arg1_mask = dev_arg1_feature.input_mask
        arg1_mask.append(dev_arg1_mask)

        dev_arg2_feature = exid_to_feature_mapping[exid][1]
        dev_arg2_mask = dev_arg2_feature.input_mask
        arg2_mask.append(dev_arg2_mask)

      # prepare bert output: [batch, total_seq_length, bert_hidden_size]
      bert_output_batch = np.asarray(bert_output_batch)
      total_seq_length = bert_output_batch.shape[1]
      assert total_seq_length == self.hp.max_arg_length * 2, \
        "Sequence length mismatch between BERT output and parameter"

      arg1 = bert_output_batch[:, :self.hp.max_arg_length, :]
      arg2 = bert_output_batch[:, self.hp.max_arg_length:, :]

      # connectives
      # since we don't care about connectives, make them 0 for now
      conn = np.zeros([self.hp.batch_size,
                       self.hp.max_arg_length,
                       bert_outputs.shape[-1]])

      per_example_loss, loss, preds, correct, acc = \
        self.sess.run(
          [self.model.per_example_loss, self.model.loss, self.model.preds,
           self.model.correct, self.model.acc],
          feed_dict={self.model.arg1          : arg1,
                     self.model.arg2          : arg2,
                     self.model.conn          : conn,
                     self.model.label         : label_ids,
                     self.model.arg1_attn_mask: arg1_mask,
                     self.model.arg2_attn_mask: arg2_mask})

      tf.logging.info(
        "[EVAL Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
          i+1, num_batches, loss, acc))

      c = Counter(preds)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))
      all_counter.update(c)

      all_loss.extend(per_example_loss)
      all_correct.extend(correct)

    tf.logging.info("[EVAL FINAL] loss: {:.3f} acc: {:.3f}".format(
      np.mean(all_loss), np.mean(all_correct)))

    for key in all_counter.keys():
      tf.logging.info(" {:3d}: {}".format(all_counter[key], self.labels[key]))

  ################################## PREDICT ###################################
  def predict(self):
    """See docstring for `run`"""

    for dataset_type in ['test', 'blind']:
      is_test_set = dataset_type == 'test'

      predict_fn = None
      if self.hp.embedding == 'bert':
        if is_test_set:
          examples = self.processor.get_test_examples(for_bert_embedding=True)
        else:
          examples = self.processor.get_blind_examples(for_bert_embedding=True)
        predict_fn = self.predict_from_vals
      else:
        if is_test_set:
          examples = self.processor.get_test_examples()
        else:
          examples = self.processor.get_blind_examples()
        predict_fn = self.predict_from_ids

      tf.logging.info(f"Inference on {dataset_type} set")
      self.processor.remove_cache_by_key(dataset_type)
      preds = predict_fn(examples, is_test_set=is_test_set)

      preds_file = \
        os.path.join(self.hp.model_dir, f"{dataset_type}_predictions.txt")

      tf.logging.info(f"Exporting {dataset_type} predictions at {preds_file}")
      with open(preds_file, 'w') as f:
        f.write("\n".join(preds))

  def predict_from_ids(self, examples, is_test_set=False):
    self.processor.remove_cache_by_key('test')

    examples = self.embedding.convert_to_ids(examples, self.l2i)
    arg1, arg2, conn, label = examples

    preds, acc = self.sess.run(
      [self.model.preds, self.model.acc],
      feed_dict={self.model.arg1 : arg1, self.model.arg2 : arg2,
                 self.model.conn : conn, self.model.label: label})

    c = Counter(preds)
    tf.logging.info("[PREDICT] acc: {:.3f} {}".format(acc, c))

    # convert label_id preds to labels
    preds_str = [self.i2l(pred) for pred in preds]
    return preds_str

  def predict_from_vals(self, examples, is_test_set=False):
    if is_test_set:
      dataset_type = "test"
    else:
      dataset_type = "blind"

    bert_outputs = self.embedding.run(examples)

    bert_outputs, exid_to_feature_mapping = bert_outputs

    example_batches = self.batchify(examples, batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)

    all_correct = []
    all_loss = []
    all_preds = []
    all_counter = Counter()

    for i, example_batch in enumerate(example_batches):
      label_ids = []
      bert_output_batch = []
      arg1_mask, arg2_mask = [], []
      for example in example_batch:
        # arg1 and arg2
        bert_output_batch.append(bert_outputs[example.exid])

        label_ids.append(self.l2i(example.label))

        exid = example.exid
        infer_arg1_feature = exid_to_feature_mapping[exid][0]
        infer_arg1_mask = infer_arg1_feature.input_mask
        arg1_mask.append(infer_arg1_mask)

        infer_arg2_feature = exid_to_feature_mapping[exid][1]
        infer_arg2_mask = infer_arg2_feature.input_mask
        arg2_mask.append(infer_arg2_mask)

      # prepare bert output: [batch, total_seq_length, bert_hidden_size]
      bert_output_batch = np.asarray(bert_output_batch)
      total_seq_length = bert_output_batch.shape[1]
      assert total_seq_length == self.hp.max_arg_length * 2, \
        "Sequence length mismatch between BERT output and parameter"

      arg1 = bert_output_batch[:, :self.hp.max_arg_length, :]
      arg2 = bert_output_batch[:, self.hp.max_arg_length:, :]

      # connectives
      # since we don't care about connectives, make them 0 for now
      conn = np.zeros([self.hp.batch_size,
                       self.hp.max_arg_length,
                       bert_outputs.shape[-1]])

      per_example_loss, loss, preds, correct, acc = \
        self.sess.run(
          [self.model.per_example_loss, self.model.loss, self.model.preds,
           self.model.correct, self.model.acc],
          feed_dict={self.model.arg1          : arg1,
                     self.model.arg2          : arg2,
                     self.model.conn          : conn,
                     self.model.label         : label_ids,
                     self.model.arg1_attn_mask: arg1_mask,
                     self.model.arg2_attn_mask: arg2_mask})

      tf.logging.info(
        "[{} Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
          dataset_type.upper(), i+1, num_batches, loss, acc))

      c = Counter(preds)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))
      all_counter.update(c)
      all_preds.extend(preds)
      all_loss.extend(per_example_loss)
      all_correct.extend(correct)

    tf.logging.info("[{} FINAL] loss: {:.3f} acc: {:.3f}".format(
      dataset_type.upper(), np.mean(all_loss), np.mean(all_correct)))

    for key in all_counter.keys():
      tf.logging.info(f" {self.labels[key]}: {all_counter[key]}")

    # convert preds to labels
    preds_str = [self.i2l(pred) for pred in all_preds]

    return preds_str
