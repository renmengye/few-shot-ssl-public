# Copyright (c) 2018 Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell,
# Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richars S. Zemel.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
"""Nearest neighbors and logistic regression baselines.

Usage:
./run_baselines_exp.py                                                    \
                             --aug              [AUGMENT 90 DEGREE]       \
                             --shuffle_episode  [SHUFFLE EPISODE]         \
                             --nclasses_eval    [NUM CLASSES EVAL]        \
                             --nclasses_train   [NUM CLASSES TRAIN]       \
                             --nshot            [NUM SHOT]                \
                             --num_eval_episode [NUM EVAL EPISODE]        \
                             --num_test         [NUM TEST]                \
                             --num_unlabel      [NUM UNLABEL]             \
                             --seed             [RANDOM SEED]             \
                             --dataset          [DATASET NAME]

Flags:

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import six
import tensorflow as tf

from fewshot.configs.config_factory import get_config
from fewshot.configs.tiered_imagenet_config import *
from fewshot.configs.mini_imagenet_config import *
from fewshot.configs.omniglot_config import *
from fewshot.data.data_factory import get_dataset
from fewshot.data.episode import Episode
from fewshot.data.tiered_imagenet import TieredImageNetDataset
from fewshot.data.mini_imagenet import MiniImageNetDataset
from fewshot.data.omniglot import OmniglotDataset
from fewshot.models.nnlib import cnn, weight_variable
from fewshot.utils import logger
from fewshot.utils.batch_iter import BatchIterator
from tqdm import tqdm

log = logger.get()


class LRModel(object):
  """A fully supervised logistic regression model for episodic learning."""

  def __init__(self, x, y, num_classes, dtype=tf.float32, learn_rate=1e-3):
    x_shape = x.get_shape()
    x_size = 1
    for ss in x_shape[1:]:
      x_size *= int(ss)
    x = tf.reshape(x, [-1, x_size])
    w_class = weight_variable(
        [x_size, num_classes],
        init_method='truncated_normal',
        dtype=tf.float32,
        init_param={'stddev': 0.01},
        name='w_class')
    b_class = weight_variable(
        [num_classes],
        init_method='constant',
        init_param={'val': 0.0},
        name='b_class')
    logits = tf.matmul(x, w_class) + b_class
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    xent = tf.reduce_mean(xent, name='xent')
    cost = xent
    cost += self._decay()
    self._cost = cost
    self._inputs = x
    self._labels = y
    self._train_op = tf.train.AdamOptimizer(learn_rate).minimize(
        cost, var_list=[w_class, b_class])
    correct = tf.equal(tf.argmax(logits, axis=1), y)
    self._acc = tf.reduce_mean(tf.cast(correct, dtype))
    self._prediction = tf.nn.softmax(logits)

  def _decay(self):
    wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    log.info('Weight decay variables')
    [log.info(x) for x in wd_losses]
    log.info('Total length: {}'.format(len(wd_losses)))
    if len(wd_losses) > 0:
      return tf.add_n(wd_losses)
    else:
      log.warning('No weight decay variables!')
      return 0.0

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels

  @property
  def cost(self):
    return self._cost

  @property
  def train_op(self):
    return self._train_op

  @property
  def acc(self):
    return self._acc

  @property
  def prediction(self):
    return self._prediction


class SupervisedModel(object):
  """A fully supervised classification model for baseline representation learning"""

  def __init__(self,
               config,
               x,
               y,
               num_classes,
               is_training=True,
               dtype=tf.float32):
    """Constructor.

    Args:
      config:
      x:
      y:
      num_classes:
    """
    h, _ = cnn(
        x,
        config.filter_size,
        strides=config.strides,
        pool_fn=[tf.nn.max_pool] * len(config.pool_fn),
        pool_size=config.pool_size,
        pool_strides=config.pool_strides,
        act_fn=[tf.nn.relu for aa in config.conv_act_fn],
        add_bias=True,
        init_std=config.conv_init_std,
        init_method=config.conv_init_method,
        wd=config.wd,
        dtype=dtype,
        batch_norm=True,
        is_training=is_training,
        ext_wts=None)
    h_shape = h.get_shape()
    h_size = 1
    for ss in h_shape[1:]:
      h_size *= int(ss)
    h = tf.reshape(h, [-1, h_size])
    w_class = weight_variable(
        [h_size, num_classes],
        init_method='truncated_normal',
        dtype=tf.float32,
        init_param={'stddev': 0.01},
        name='w_class')
    b_class = weight_variable(
        [num_classes],
        init_method='constant',
        init_param={'val': 0.0},
        name='b_class')
    self._feature = h
    logits = tf.matmul(h, w_class) + b_class
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y)
    xent = tf.reduce_mean(xent, name='xent')
    cost = xent
    cost += self._decay()
    self._cost = cost
    self._inputs = x
    self._labels = y
    global_step = tf.get_variable(
        'global_step', shape=[], dtype=tf.int64, trainable=False)
    # Learning rate decay.
    learn_rate = tf.train.piecewise_constant(
        global_step, list(np.array(config.lr_decay_steps).astype(np.int64)),
        [config.learn_rate] + list(config.lr_list))
    self._learn_rate = learn_rate
    self._train_op = tf.train.AdamOptimizer(learn_rate).minimize(
        cost, global_step=global_step)

    correct = tf.equal(tf.argmax(logits, axis=1), y)
    self._acc = tf.reduce_mean(tf.cast(correct, dtype))

  def _decay(self):
    wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    log.info('Weight decay variables')
    [log.info(x) for x in wd_losses]
    log.info('Total length: {}'.format(len(wd_losses)))
    if len(wd_losses) > 0:
      return tf.add_n(wd_losses)
    else:
      log.warning('No weight decay variables!')
      return 0.0

  @property
  def inputs(self):
    return self._inputs

  @property
  def labels(self):
    return self._labels

  @property
  def cost(self):
    return self._cost

  @property
  def train_op(self):
    return self._train_op

  @property
  def acc(self):
    return self._acc

  @property
  def feature(self):
    return self._feature

  @property
  def learn_rate(self):
    return self._learn_rate


def get_exp_logger(sess, log_folder):
  """Gets a TensorBoard logger."""
  with tf.name_scope('Summary'):
    writer = tf.summary.FileWriter(os.path.join(log_folder, 'logs'), sess.graph)

  class ExperimentLogger():

    def log(self, name, niter, value):
      summary = tf.Summary()
      summary.value.add(tag=name, simple_value=value)
      writer.add_summary(summary, niter)

    def flush(self):
      """Flushes results to disk."""
      writer.flush()

    def close(self):
      """Closes writer."""
      writer.close()

  return ExperimentLogger()


def supervised_pretrain(sess,
                        model,
                        train_data,
                        num_steps,
                        num_eval_steps=10,
                        batch_size=100,
                        logging_fn=None):
  """Pretrain a supervised model on the labeled split of the training data to get a reasonable
  embedding model for baselines.

  Args:
    sess: TensorFlow session object.
    model: SupervisedModel object.
    train_data: Training dataset object.
    test_data: Testing dataset object.
    num_steps: Int. Number of training steps.
  """
  train_iter = BatchIterator(
      train_data.get_size(),
      batch_size=batch_size,
      cycle=True,
      shuffle=True,
      get_fn=train_data.get_batch_idx,
      log_epoch=-1)
  train_eval_iter = BatchIterator(
      train_data.get_size(),
      batch_size=batch_size,
      cycle=True,
      shuffle=True,
      get_fn=train_data.get_batch_idx,
      log_epoch=-1)
  test_iter = BatchIterator(
      train_data.get_size(),
      batch_size=batch_size,
      cycle=True,
      shuffle=True,
      get_fn=train_data.get_batch_idx_test,
      log_epoch=-1)

  sess.run(tf.global_variables_initializer())

  it = tqdm(six.moves.xrange(num_steps), ncols=0)
  for ii in it:
    x_train, y_train = train_iter.next()
    sess.run(
        [model.train_op],
        feed_dict={
            model.inputs: x_train,
            model.labels: y_train
        })
    if (ii + 1) % 100 == 0 or ii == 0:
      train_cost = 0.0
      train_acc = 0.0
      for jj in six.moves.xrange(num_eval_steps):
        x_train, y_train = train_eval_iter.next()
        cost_, acc_ = sess.run(
            [model.cost, model.acc],
            feed_dict={
                model.inputs: x_train,
                model.labels: y_train
            })
        train_cost += cost_ / num_eval_steps
        train_acc += acc_ / num_eval_steps

      test_cost = 0.0
      test_acc = 0.0
      for jj in six.moves.xrange(num_eval_steps):
        x_train, y_train = test_iter.next()
        cost_, acc_ = sess.run(
            [model.cost, model.acc],
            feed_dict={
                model.inputs: x_train,
                model.labels: y_train
            })
        test_cost += cost_ / num_eval_steps
        test_acc += acc_ / num_eval_steps

      learn_rate = sess.run(model.learn_rate)
      if logging_fn is not None:
        logging_fn(
            ii + 1, {
                'train_cost': train_cost,
                'train_acc': train_acc,
                'test_cost': test_cost,
                'test_acc': test_acc,
                'learn_rate': learn_rate
            })
      it.set_postfix(
          ce='{:.3e}'.format(train_cost),
          train_acc='{:.3f}%'.format(train_acc * 100),
          test_acc='{:.3f}%'.format(test_acc * 100),
          lr='{:.3e}'.format(learn_rate))


def preprocess_batch(batch):
  if len(batch.x_train.shape) == 4:
    x_train = np.expand_dims(batch.x_train, 0)
    y_train = np.expand_dims(batch.y_train, 0)
    x_test = np.expand_dims(batch.x_test, 0)
    y_test = np.expand_dims(batch.y_test, 0)
    if batch.x_unlabel is not None:
      x_unlabel = np.expand_dims(batch.x_unlabel, 0)
    else:
      x_unlabel = None
    if hasattr(batch, 'y_unlabel') and batch.y_unlabel is not None:
      y_unlabel = np.expand_dims(batch.y_unlabel, 0)
    else:
      y_unlabel = None

    return Episode(
        x_train,
        y_train,
        x_test,
        y_test,
        x_unlabel=x_unlabel,
        y_unlabel=y_unlabel,
        y_train_str=batch.y_train_str,
        y_test_str=batch.y_test_str)
  else:
    return batch


def get_nn_fit(x_train, y_train, x_test, k=1):
  """Fit a nearest neighbor classifier.

  Args:
    x_train: Training inputs. [N, H, W, C].
    y_train: Training integer class labels. [N].
    x_test: Test inputs. [N, H, W, C].
    k: Int. Number of nearest neighbors to consider. Default 1.

  Returns:
    y_pred: Test prediction integer class labels. [N].
  """
  nbatches = x_train.shape[0]
  y_pred = np.zeros([x_test.shape[0], x_test.shape[1]])
  for ii in six.moves.xrange(nbatches):
    x_train_ = x_train.reshape([x_train[ii].shape[0], -1])
    y_train_ = y_train.reshape([x_train[ii].shape[0]])
    x_test_ = x_test.reshape([x_test[ii].shape[0], -1])
    x_train_ = np.expand_dims(x_train_, 1)
    x_test_ = np.expand_dims(x_test_, 0)
    pairdist = ((x_train_ - x_test_)**2).sum(axis=-1)
    assert k == 1, 'Only support k=1 for now'
    min_idx = np.argmin(pairdist, axis=0)
    sort_idx = np.argsort(pairdist, axis=0)
    y_pred[ii] = y_train[ii, min_idx]
  return y_pred


def run_nn(sess, meta_dataset, num_episodes=600, emb_model=None):
  """Nearest neighbor baselines."""
  ncorr = 0
  ntotal = 0

  acc_list = []
  for neval in tqdm(six.moves.xrange(num_episodes), ncols=0):
    dataset = meta_dataset.next()
    batch = dataset.next_batch()
    batch = preprocess_batch(batch)
    if emb_model is not None:
      x_train = sess.run(
          emb_model.feature,
          feed_dict={
              emb_model.inputs: np.squeeze(batch.x_train, axis=0)
          })
      x_test = sess.run(
          emb_model.feature,
          feed_dict={
              emb_model.inputs: np.squeeze(batch.x_test, axis=0)
          })
      x_train = np.expand_dims(x_train, axis=0)
      x_test = np.expand_dims(x_test, axis=0)
    else:
      x_train = batch.x_train
      x_test = batch.x_test
    y_pred = get_nn_fit(x_train, batch.y_train, x_test)
    ncorr_ = np.equal(y_pred, batch.y_test).astype(np.float32)
    ncorr += ncorr_.sum()
    ntotal += y_pred.size
    acc_list.append(ncorr_.sum() / float(y_pred.size))
  meta_dataset.reset()
  acc_list = np.array(acc_list)
  print('Acc', ncorr / float(ntotal))
  print('Std', acc_list.std())
  print('95 CI', acc_list.std() * 1.96 / np.sqrt(float(num_episodes)))


def get_lr_fit(sess, model, x_train, y_train, x_test, num_steps=100):
  """Fit a multi-class logistic regression classifier.
  Args:
    x_train: [N, D]. Training data.
    y_train: [N]. Training label, integer classes.
    x_test: [M, D]. Test data.

  Returns:
    y_pred: [M]. Integer class prediction of test data.
  """

  nbatches = x_train.shape[0]
  y_pred = np.zeros([x_test.shape[0], x_test.shape[1]])
  for ii in six.moves.xrange(nbatches):
    x_train_ = x_train[ii].reshape([x_train[ii].shape[0], -1])
    x_test_ = x_test[ii].reshape([x_test[ii].shape[0], -1])
    y_train_ = y_train[ii]

    # Reinitialize variables for a new episode.
    var_to_init = list(
        filter(lambda x: 'LRModel' in x.name, tf.global_variables()))
    sess.run(tf.variables_initializer(var_to_init))

    # Run LR training.
    for step in six.moves.xrange(num_steps):
      cost, acc, _ = sess.run(
          [model.cost, model.acc, model.train_op],
          feed_dict={
              model.inputs: x_train_,
              model.labels: y_train_
          })
    y_pred[ii] = np.argmax(
        sess.run(model.prediction, feed_dict={
            model.inputs: x_test_
        }), axis=-1)
  return y_pred


def run_lr(sess,
           meta_dataset,
           input_shape,
           feature_shape,
           num_episodes=600,
           num_classes=5,
           emb_model=None):
  """Logistic regression baselines."""

  def get_lr_model(x_shape=[None, 28, 28, 1], learn_rate=1e-3):
    with log.verbose_level(2):
      x = tf.placeholder(tf.float32, x_shape, name='x')
      y = tf.placeholder(tf.int64, [None], name='y')
      with tf.variable_scope('LRModel'):
        lr_model = LRModel(x, y, num_classes, learn_rate=learn_rate)
    return lr_model

  ncorr = 0
  ntotal = 0
  if emb_model is not None:
    model = get_lr_model(x_shape=[None] + feature_shape, learn_rate=1e-2)
    num_steps = 200
    # I tried 2000 here doesn't help.
  else:
    model = get_lr_model(x_shape=[None] + input_shape, learn_rate=1e-3)
    num_steps = 200

  acc_list = []
  for neval in tqdm(six.moves.xrange(num_episodes), ncols=0):
    dataset = meta_dataset.next()
    batch = dataset.next_batch()
    batch = preprocess_batch(batch)

    if emb_model is not None:
      x_train = sess.run(
          emb_model.feature,
          feed_dict={
              emb_model.inputs: np.squeeze(batch.x_train, axis=0)
          })
      x_test = sess.run(
          emb_model.feature,
          feed_dict={
              emb_model.inputs: np.squeeze(batch.x_test, axis=0)
          })
      x_train = np.expand_dims(x_train, axis=0)
      x_test = np.expand_dims(x_test, axis=0)
    else:
      x_train = batch.x_train
      x_test = batch.x_test

    y_pred = get_lr_fit(
        sess, model, x_train, batch.y_train, x_test, num_steps=num_steps)
    ncorr_ = np.equal(y_pred, batch.y_test).astype(np.float32)
    ncorr += ncorr_.sum()
    ntotal += y_pred.size
    acc_list.append(ncorr_.sum() / float(y_pred.size))
  meta_dataset.reset()
  acc_list = np.array(acc_list)
  print('Acc', ncorr / float(ntotal))
  print('Std', acc_list.std())
  print('95 CI', acc_list.std() * 1.96 / np.sqrt(float(num_episodes)))


def main():
  # ------------------------------------------------------------------------
  # Flags.
  if FLAGS.num_test == -1 and (FLAGS.dataset == "tiered-imagenet" or
                               FLAGS.dataset == 'mini-imagenet'):
    num_test = 5
  else:
    num_test = FLAGS.num_test
  nclasses_train = FLAGS.nclasses_train
  nclasses_eval = FLAGS.nclasses_eval

  # Whether doing 90 degree augmentation.
  if 'mini-imagenet' in FLAGS.dataset or 'tiered-imagenet' in FLAGS.dataset:
    _aug_90 = False
    input_shape = [84, 84, 3]
    feature_shape = [1600]
  else:
    _aug_90 = True
    input_shape = [28, 28, 1]
    feature_shape = [64]

  nshot = FLAGS.nshot
  dataset = FLAGS.dataset

  meta_train_dataset = get_dataset(
      FLAGS.dataset,
      'train',
      nclasses_train,
      nshot,
      num_test=num_test,
      aug_90=_aug_90,
      num_unlabel=FLAGS.num_unlabel,
      shuffle_episode=FLAGS.shuffle_episode,
      seed=FLAGS.seed)
  meta_val_dataset = get_dataset(
      FLAGS.dataset,
      'val',
      nclasses_eval,
      nshot,
      num_test=num_test,
      aug_90=_aug_90,
      num_unlabel=FLAGS.num_unlabel,
      shuffle_episode=FLAGS.shuffle_episode,
      seed=FLAGS.seed)
  meta_test_dataset = get_dataset(
      FLAGS.dataset,
      "test",
      nclasses_eval,
      nshot,
      num_test=num_test,
      aug_90=_aug_90,
      num_unlabel=FLAGS.num_unlabel,
      shuffle_episode=FLAGS.shuffle_episode,
      seed=FLAGS.seed)

  # ------------------------------------------------------------------------
  # Get embedding model.
  def get_emb_model(config, dataset, is_training=True):
    log.info('Building embedding model')
    with log.verbose_level(2):
      x = tf.placeholder(
          tf.float32, [None, config.height, config.width, config.num_channel],
          name='x')
      y = tf.placeholder(tf.int64, [None], name='y')
      with tf.variable_scope('EmbeddingModel'):
        emb_model = SupervisedModel(
            config, x, y, dataset.num_classes, is_training=is_training)
      log.info('Training embedding model in fully supervised mode')
    return emb_model

  # Get supervised training logging function.
  def get_logging_fn(sess, log_folder):
    exp_logger = get_exp_logger(sess, log_folder)

    def _logging_fn(niter, data):
      # log.info(
      #     'Step {} Train Cost {:.3e} Train Acc {:.3f} Test Cost {:.3e} Test Acc {:.3f}'.
      #     format(niter, data['train_cost'], data['train_acc'] * 100.0, data[
      #         'test_cost'], data['test_acc'] * 100.0))
      for key in data:
        exp_logger.log(key, niter, data[key])
      exp_logger.flush()

    return _logging_fn

  # ------------------------------------------------------------------------
  # Pretrain an embedding model with train dataset (for new version of the paper).
  ckpt_train = os.path.join('results', dataset, 'supv_emb_model_train',
                            'model.ckpt')
  log_folder_train = os.path.join('results', dataset, 'supv_emb_model_train')
  ckpt_dir = os.path.dirname(ckpt_train)
  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
  if not os.path.exists(ckpt_train + '.meta'):
    with tf.Graph().as_default(), tf.Session() as sess:
      config = get_config(dataset, 'basic-pretrain')
      emb_model_train = get_emb_model(config, meta_train_dataset)
      logging_fn = get_logging_fn(sess, log_folder_train)
      supervised_pretrain(
          sess,
          emb_model_train,
          meta_train_dataset,
          num_steps=config.max_train_steps,
          logging_fn=logging_fn)

      # Save model to a checkpoint.
      saver = tf.train.Saver()
      saver.save(sess, ckpt_train)
  else:
    log.info('Checkpoint found. Skip pretraining.')

  # ------------------------------------------------------------------------
  # Run nearest neighbor in the pixel space.
  with tf.Graph().as_default(), tf.Session() as sess:
    log.info('Nearest neighbor baseline in the pixel space')
    run_nn(sess, meta_test_dataset, num_episodes=FLAGS.num_eval_episode)

  # ------------------------------------------------------------------------
  # Run logistic regression in the pixel space.
  with tf.Graph().as_default(), tf.Session() as sess:
    log.info('Logistic regression in the pixel space')
    run_lr(
        sess,
        meta_test_dataset,
        input_shape,
        feature_shape,
        num_episodes=FLAGS.num_eval_episode)

  # ------------------------------------------------------------------------
  # Run nearest neighbor in the embedding space, using train model.
  with tf.Graph().as_default(), tf.Session() as sess:
    log.info(
        'Nearest neighbor baseline in feature space, pretrained features, train'
    )
    config = get_config(dataset, 'basic-pretrain')
    emb_model_train = get_emb_model(
        config, meta_train_dataset, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_train)
    run_nn(
        sess,
        meta_test_dataset,
        emb_model=emb_model_train,
        num_episodes=FLAGS.num_eval_episode)

  # ------------------------------------------------------------------------
  # Run nearest neighbor in the embedding space, using train model, with random features.
  with tf.Graph().as_default(), tf.Session() as sess:
    log.info('Nearest neighbor baseline in feature space, random features')
    config = get_config(dataset, 'basic-pretrain')
    emb_model_train = get_emb_model(
        config, meta_train_dataset, is_training=False)
    sess.run(tf.global_variables_initializer())
    run_nn(
        sess,
        meta_test_dataset,
        emb_model=emb_model_train,
        num_episodes=FLAGS.num_eval_episode)

  # ------------------------------------------------------------------------
  # Run logistic regression in the embedding space, using train model.
  with tf.Graph().as_default(), tf.Session() as sess:
    log.info(
        'Logistic regression in the feature space, pretrained features, train')
    config = get_config(dataset, 'basic-pretrain')
    emb_model_train = get_emb_model(
        config, meta_train_dataset, is_training=False)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_train)
    run_lr(
        sess,
        meta_test_dataset,
        input_shape,
        feature_shape,
        num_episodes=FLAGS.num_eval_episode,
        emb_model=emb_model_train)

  # ------------------------------------------------------------------------
  # Run logistic regression in the embedding space, using train model, with random features.
  with tf.Graph().as_default(), tf.Session() as sess:
    log.info('Logistic regression in the feature space, random features')
    config = get_config(dataset, 'basic-pretrain')
    emb_model_train = get_emb_model(
        config, meta_train_dataset, is_training=False)
    sess.run(tf.global_variables_initializer())
    run_lr(
        sess,
        meta_test_dataset,
        input_shape,
        feature_shape,
        num_episodes=FLAGS.num_eval_episode,
        emb_model=emb_model_train)


if __name__ == '__main__':
  flags = tf.flags
  FLAGS = tf.flags.FLAGS
  flags.DEFINE_bool("aug", True, "Whether perform 90 degree data augmentation")
  flags.DEFINE_bool("shuffle_episode", False,
                    "Whether to shuffle the sequence order")
  flags.DEFINE_bool("final_eval", False, "Final eval for tieredImageNet")
  flags.DEFINE_integer("nclasses_eval", 5, "Number of classes for testing")
  flags.DEFINE_integer("nclasses_train", 5, "Number of classes for training")
  flags.DEFINE_integer("nshot", 1, "nshot")
  flags.DEFINE_integer("num_eval_episode", 600, "Number of evaluation episodes")
  flags.DEFINE_integer("num_test", -1, "Number of test images per episode")
  flags.DEFINE_integer("num_unlabel", 5, "Number of unlabeled for training")
  flags.DEFINE_integer("seed", 0, "Random seed")
  flags.DEFINE_string("dataset", "omniglot", "Dataset name")
  main()
