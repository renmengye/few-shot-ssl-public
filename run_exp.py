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
"""Runs a single experiment.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
  python run_exp.py --data_root {DATA_ROOT}            \
                    --dataset {DATASET}                \
                    --label_ratio {LABEL_RATIO}        \
                    --model {MODEL}                    \
                    --results {SAVE_CKPT_FOLDER}       \
                    [--disable_distractor]             \
                    [--eval]                           \
                    [--num_unlabel {NUM_UNLABEL}]      \
                    [--num_test {NUM_TEST}]            \
                    [--pretrain {MODEL_ID}]            \
                    [--use_test]


Example:
  # To train a model for Omniglot:
  python run_exp.py --data_root /data/ \
                    --dataset omniglot \
                    --label_ratio 0.1  \
                    --model basic      \
                    --results /ckpt/

  # To run evaluation, grab the model ID from training:
  python run_exp.py --data_root /data/ \
                    --dataset omniglot \
                    --label_ratio 0.1  \
                    --model basic      \
                    --results /ckpt/   \
                    --eval             \
                    --pretrain {ID}

Flags:
  --data_root: String. Path to the root for storing all datasets.
  --dataset: String. Name of the dataset. Options: `omniglot`, `mini-imagenet`, `tiered-imagenet
  --disable_distractor: Whether to remove all distractor classes in the unlabeled images.
  --eval: Bool. Whether to run evaluation only.
  --label_ratio: Float. Proportion of the training data used for the labelled portion.
  --num_test: Int. Number of query images per class in each episode.
  --num_unlabel: Int. Number of unlabeled images per class in each episode.
  --pretrain: String. Model ID obtained from training.
  --results: String. Path to the folder for storing all checkpoints.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import json
import os
import numpy as np
import six
import tensorflow as tf

from fewshot.configs.config_factory import get_config
from fewshot.configs.mini_imagenet_config import *
from fewshot.configs.omniglot_config import *
from fewshot.configs.tiered_imagenet_config import *
from fewshot.data.data_factory import get_concurrent_iterator
from fewshot.data.data_factory import get_dataset
from fewshot.data.episode import Episode
from fewshot.data.mini_imagenet import MiniImageNetDataset
from fewshot.data.omniglot import OmniglotDataset
from fewshot.data.tiered_imagenet import TieredImageNetDataset
from fewshot.models.basic_model import BasicModel
from fewshot.models.kmeans_refine_mask_model import KMeansRefineMaskModel
from fewshot.models.kmeans_refine_model import KMeansRefineModel
from fewshot.models.kmeans_refine_radius_model import KMeansRefineRadiusModel
from fewshot.models.measure import batch_apk
from fewshot.models.model_factory import get_model
from fewshot.utils import logger
from fewshot.utils.experiment_logger import ExperimentLogger
from fewshot.utils.lr_schedule import FixedLearnRateScheduler
from tqdm import tqdm

log = logger.get()

flags = tf.flags
flags.DEFINE_bool("eval", False, "Whether to only run evaluation")
flags.DEFINE_bool("use_test", False, "Use the test set or not")
flags.DEFINE_float("learn_rate", None, "Start learning rate")
flags.DEFINE_integer("nclasses_eval", 5, "Number of classes for testing")
flags.DEFINE_integer("nclasses_train", 5, "Number of classes for training")
flags.DEFINE_integer("nshot", 1, "nshot")
flags.DEFINE_integer("num_eval_episode", 600, "Number of evaluation episodes")
flags.DEFINE_integer("num_test", -1, "Number of test images per episode")
flags.DEFINE_integer("num_unlabel", 5, "Number of unlabeled for training")
flags.DEFINE_integer("seed", 0, "Random seed")
flags.DEFINE_string("dataset", "omniglot", "Dataset name")
flags.DEFINE_string("model", "basic", "Model name")
flags.DEFINE_string("pretrain", None, "Model pretrain path")
flags.DEFINE_string("results", "./results", "Checkpoint save path")
FLAGS = tf.flags.FLAGS
log = logger.get()


def _get_model(config, nclasses_train, nclasses_eval):
  with tf.name_scope("MetaTrain"):
    with tf.variable_scope("Model"):
      m = get_model(
          config.model_class,
          config,
          nclasses_train,
          is_training=True,
          nshot=FLAGS.nshot)
  with tf.name_scope("MetaValid"):
    with tf.variable_scope("Model", reuse=True):
      mvalid = get_model(
          config.model_class,
          config,
          nclasses_eval,
          is_training=False,
          nshot=FLAGS.nshot)
  return m, mvalid


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


def evaluate(sess, model, meta_dataset, num_episodes=FLAGS.num_eval_episode):
  ncorr = 0
  ntotal = 0
  all_acc = []
  for neval in tqdm(six.moves.xrange(num_episodes), desc="evaluation", ncols=0):
    dataset = meta_dataset.next()
    batch = dataset.next_batch()
    batch = preprocess_batch(batch)
    feed_dict = {
        model.x_train: batch.x_train,
        model.y_train: batch.y_train,
        model.x_test: batch.x_test,
    }
    if hasattr(model, '_x_unlabel'):
      if batch.x_unlabel is not None:
        feed_dict[model.x_unlabel] = batch.x_unlabel
      else:
        feed_dict[model.x_unlabel] = batch.x_test
    outputs = [model.prediction]
    results = sess.run(outputs, feed_dict=feed_dict)
    y_pred = results[0]
    y_pred = np.argmax(y_pred, axis=2)
    _ncorr = np.equal(y_pred, batch.y_test).sum()
    ncorr += _ncorr
    ntotal += batch.y_test.size
    all_acc.append(_ncorr / float(batch.y_test.size))
  acc = ncorr / float(ntotal)
  return {'acc': acc, 'acc_ci': np.std(all_acc) * 1.96 / np.sqrt(num_episodes)}


def gen_id(config):
  return "{}_{}-{:03d}".format(config.name,
                               datetime.datetime.now().isoformat(chr(
                                   ord("-"))).replace(":", "-").replace(
                                       ".", "-"), int(np.random.rand() * 1000))


def save(sess, saver, niter, save_folder):
  if not os.path.exists(save_folder):
    os.makedirs(save_folder)
  saver.save(sess, os.path.join(save_folder, "model.ckpt"), global_step=niter)


def save_config(config, save_folder):
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
  config_file = os.path.join(save_folder, "conf.json")
  with open(config_file, "w") as f:
    f.write(json.dumps(dict(config.__dict__)))


def train(sess,
          config,
          model,
          meta_dataset,
          mvalid=None,
          meta_val_dataset=None,
          log_results=True,
          run_eval=True,
          exp_id=None):
  lr_scheduler = FixedLearnRateScheduler(
      sess,
      model,
      config.learn_rate,
      config.lr_decay_steps,
      lr_list=config.lr_list)

  if exp_id is None:
    exp_id = gen_id(config)

  saver = tf.train.Saver()
  save_folder = os.path.join(FLAGS.results, exp_id)
  save_config(config, save_folder)
  if log_results:
    logs_folder = os.path.join("logs", exp_id)
    exp_logger = ExperimentLogger(logs_folder)
  it = tqdm(six.moves.xrange(config.max_train_steps), desc=exp_id, ncols=0)

  trn_acc = 0.0
  val_acc = 0.0
  lr = lr_scheduler.lr
  for niter in it:
    lr_scheduler.step(niter)
    dataset = meta_dataset.next()
    batch = dataset.next_batch()
    batch = preprocess_batch(batch)

    feed_dict = {
        model.x_train: batch.x_train,
        model.y_train: batch.y_train,
        model.x_test: batch.x_test,
        model.y_test: batch.y_test
    }
    if hasattr(model, '_x_unlabel'):
      if batch.x_unlabel is not None:
        feed_dict[model.x_unlabel] = batch.x_unlabel
      else:
        feed_dict[model.x_unlabel] = batch.x_test

    loss_val, y_pred, _ = sess.run(
        [model.loss, model.prediction, model.train_op], feed_dict=feed_dict)

    if (niter + 1) % config.steps_per_valid == 0 and run_eval:
      train_results = evaluate(sess, mvalid, meta_dataset)
      if log_results:
        exp_logger.log_train_acc(niter, train_results['acc'])
        exp_logger.log_learn_rate(niter, lr_scheduler.lr)
        lr = lr_scheduler.lr
        trn_acc = train_results['acc']

      if mvalid is not None:
        val_results = evaluate(sess, mvalid, meta_val_dataset)

        if log_results:
          exp_logger.log_valid_acc(niter, val_results['acc'])
          exp_logger.log_learn_rate(niter, lr_scheduler.lr)
          val_acc = val_results['acc']
          it.set_postfix()
          meta_val_dataset.reset()

    if (niter + 1) % config.steps_per_log == 0 and log_results:
      exp_logger.log_train_ce(niter + 1, loss_val)
      it.set_postfix(
          ce='{:.3e}'.format(loss_val),
          trn_acc='{:.3f}'.format(trn_acc * 100.0),
          val_acc='{:.3f}'.format(val_acc * 100.0),
          lr='{:.3e}'.format(lr))

    if (niter + 1) % config.steps_per_save == 0:
      save(sess, saver, niter, save_folder)
  return exp_id


def main():
  if FLAGS.num_test == -1 and (FLAGS.dataset == "tiered-imagenet" or
                               FLAGS.dataset == 'mini-imagenet'):
    num_test = 5
  else:
    num_test = FLAGS.num_test
  config = get_config(FLAGS.dataset, FLAGS.model)
  nclasses_train = FLAGS.nclasses_train
  nclasses_eval = FLAGS.nclasses_eval

  # Which training split to use.
  train_split_name = 'train'
  if FLAGS.use_test:
    log.info('Using the test set')
    test_split_name = 'test'
  else:
    log.info('Not using the test set, using val')
    test_split_name = 'val'

  log.info('Use split `{}` for training'.format(train_split_name))

  # Whether doing 90 degree augmentation.
  if 'mini-imagenet' in FLAGS.dataset or 'tiered-imagenet' in FLAGS.dataset:
    _aug_90 = False
  else:
    _aug_90 = True

  nshot = FLAGS.nshot
  meta_train_dataset = get_dataset(
      FLAGS.dataset,
      train_split_name,
      nclasses_train,
      nshot,
      num_test=num_test,
      aug_90=_aug_90,
      num_unlabel=FLAGS.num_unlabel,
      shuffle_episode=False,
      seed=FLAGS.seed)
  meta_train_dataset = get_concurrent_iterator(
      meta_train_dataset, max_queue_size=100, num_threads=5)
  meta_test_dataset = get_dataset(
      FLAGS.dataset,
      test_split_name,
      nclasses_eval,
      nshot,
      num_test=num_test,
      aug_90=_aug_90,
      num_unlabel=FLAGS.num_unlabel,
      shuffle_episode=False,
      label_ratio=1,
      seed=FLAGS.seed)
  meta_test_dataset = get_concurrent_iterator(
      meta_test_dataset, max_queue_size=100, num_threads=5)
  m, mvalid = _get_model(config, nclasses_train, nclasses_eval)

  sconfig = tf.ConfigProto()
  sconfig.gpu_options.allow_growth = True
  with tf.Session(config=sconfig) as sess:
    if FLAGS.pretrain is not None:
      ckpt = tf.train.latest_checkpoint(
          os.path.join(FLAGS.results, FLAGS.pretrain))
      saver = tf.train.Saver()
      saver.restore(sess, ckpt)
    else:
      sess.run(tf.global_variables_initializer())
      train(sess, config, m, meta_train_dataset, mvalid, meta_test_dataset)

    results_train = evaluate(sess, mvalid, meta_train_dataset)
    results_test = evaluate(sess, mvalid, meta_test_dataset)

    log.info("Final train acc {:.3f}% ({:.3f}%)".format(
        results_train['acc'] * 100.0, results_train['acc_ci'] * 100.0))
    log.info("Final test acc {:.3f}% ({:.3f}%)".format(
        results_test['acc'] * 100.0, results_test['acc_ci'] * 100.0))


if __name__ == "__main__":
  main()
