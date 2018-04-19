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
"""Runs multiple experiments over random labeled/unlabeled splits of the dataset.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:

Example:
  python run_multi_exp.py --data_root /data/ \
                          --dataset omniglot \
                          --label_ratio 0.1  \
                          --model basic


Flags: Same set of flags as `run_exp.py`.

"""

from __future__ import division, print_function

import numpy as np
import os
import six
import tensorflow as tf

from collections import namedtuple

flags = tf.flags
flags.DEFINE_bool("eval_train", False, "Whether to evaluate training set")
flags.DEFINE_string("num_unlabel_list", None, "Number of unlabel items")

from fewshot.utils import logger
from run_exp import _get_model
from run_exp import evaluate
from run_exp import get_config
from run_exp import get_dataset
from run_exp import train

FLAGS = tf.flags.FLAGS

if 'imagenet' in FLAGS.dataset:
  if FLAGS.num_unlabel_list is None:
    NUM_UNLABEL_LIST = '0,1,2,5,10,15,20,25'
  else:
    NUM_UNLABEL_LIST = FLAGS.num_unlabel_list
  NUM_RUN = 10
else:
  if FLAGS.num_unlabel_list is None:
    NUM_UNLABEL_LIST = '0,1,2,5,10'
  else:
    NUM_UNLABEL_LIST = FLAGS.num_unlabel_list
  NUM_RUN = 10
log = logger.get()


def gen_id(config):
  import datetime
  dtstr = datetime.datetime.now().isoformat(chr(ord("-"))).replace(":",
                                                                   "-").replace(
                                                                       ".", "-")
  return "{}_{}".format(config.name, dtstr)


def run_one(dataset, model, seed, pretrain_id, exp_id):
  log.info("Random seed = {}".format(seed))
  config = get_config(dataset, model)
  nclasses_train = FLAGS.nclasses_train
  nclasses_eval = FLAGS.nclasses_eval
  train_split_name = 'train'

  if FLAGS.use_test:
    log.info('Using the test set')
    test_split_name = 'test'
  else:
    log.info('Not using the test set, using val')
    test_split_name = 'val'

  if dataset in ['mini-imagenet', 'tiered-imagenet']:
    _aug_90 = False
    num_test_test = 20
    num_train_test = 5
  else:
    _aug_90 = True
    num_test_test = -1
    num_train_test = -1

  meta_train_dataset = get_dataset(
      dataset,
      train_split_name,
      nclasses_train,
      FLAGS.nshot,
      num_test=num_train_test,
      aug_90=_aug_90,
      num_unlabel=FLAGS.num_unlabel,
      shuffle_episode=False,
      seed=seed)

  meta_test_dataset = get_dataset(
      dataset,
      test_split_name,
      nclasses_eval,
      FLAGS.nshot,
      num_test=num_test_test,
      aug_90=_aug_90,
      num_unlabel=5,
      shuffle_episode=False,
      label_ratio=1,
      seed=seed)

  sconfig = tf.ConfigProto()
  sconfig.gpu_options.allow_growth = True
  with tf.Session(config=sconfig) as sess:
    tf.set_random_seed(seed)
    with log.verbose_level(2):
      m, mvalid = _get_model(config, nclasses_train, nclasses_eval)
    if pretrain_id is not None:
      ckpt = tf.train.latest_checkpoint(
          os.path.join(FLAGS.results, pretrain_id))
      saver = tf.train.Saver()
      saver.restore(sess, ckpt)
    else:
      sess.run(tf.global_variables_initializer())

    if not FLAGS.eval:
      exp_id_ = exp_id + "-{:05d}".format(seed)
      train(
          sess,
          config,
          m,
          meta_train_dataset,
          mvalid,
          meta_test_dataset,
          log_results=False,
          run_eval=False,
          exp_id=exp_id_)
    else:
      exp_id_ = None

    if FLAGS.eval_train:
      train_results = evaluate(sess, mvalid, meta_train_dataset)
      log.info("Final train acc {:.3f}% ({:.3f}%)".format(
          train_results['acc'] * 100.0, train_results['acc_ci'] * 100.0))
    else:
      train_results = None

    num_unlabel_list = [int(nn) for nn in NUM_UNLABEL_LIST.split(',')]
    test_results_list = []
    for nn in num_unlabel_list:

      if dataset == 'mini-imagenet':
        AL_Instance = namedtuple(
            'AL_Instance', 'n_class, n_distractor, k_train, k_test, k_unlbl')
        new_al_instance = AL_Instance(
            n_class=meta_test_dataset.al_instance.n_class,
            n_distractor=meta_test_dataset.al_instance.n_distractor,
            k_train=meta_test_dataset.al_instance.k_train,
            k_test=meta_test_dataset.al_instance.k_test,
            k_unlbl=nn)
        meta_test_dataset.al_instance = new_al_instance
      else:
        meta_test_dataset._num_unlabel = nn

      meta_test_dataset.reset()
      _test_results = evaluate(sess, mvalid, meta_test_dataset)
      test_results_list.append(_test_results)
      log.info("Final test acc {:.3f}% ({:.3f}%)".format(
          _test_results['acc'] * 100.0, _test_results['acc_ci'] * 100.0))

  return train_results, test_results_list, exp_id_, num_unlabel_list


def calc_avg(number):
  number_ = np.array(number)
  return np.mean(number_), np.std(number_)


def collect(results):
  acc = [rr['acc'] for rr in results]
  return calc_avg(acc)


def main():
  rnd = np.random.RandomState(0)

  # Set up pretrain ID list.
  if FLAGS.pretrain is not None:
    num_runs = NUM_RUN
    pretrain_ids = [
        FLAGS.pretrain + '-{:05d}'.format(1001 * ii)
        for ii in six.moves.xrange(num_runs)
    ]
  else:
    pretrain_ids = [None] * NUM_RUN
    num_runs = NUM_RUN

  all_train_results = []
  all_test_results = []
  exp_ids = []
  seed_list = []
  config = get_config(FLAGS.dataset, FLAGS.model)
  exp_id_root = gen_id(config)

  for ii, pid in enumerate(pretrain_ids):
    log.info("Run {} out of {}".format(ii + 1, NUM_RUN))
    with tf.Graph().as_default():
      _seed = 1001 * ii
      train_results, test_results_list, exp_id, num_unlabel_list = run_one(
          FLAGS.dataset, FLAGS.model, _seed, pid, exp_id_root)
      all_train_results.append(train_results)
      all_test_results.append(test_results_list)
      exp_ids.append(exp_id)
      seed_list.append(_seed)

  if FLAGS.eval_train:
    trn_acc = collect(all_train_results)
    log.info('Train Acc = {:.3f} ({:.3f})'.format(trn_acc[0] * 100.0,
                                                  trn_acc[1] * 100.0))
  for ii in range(len(num_unlabel_list)):
    _all_test_results = []
    for vr in all_test_results:
      _all_test_results.append(vr[ii])
    _test_acc = collect(_all_test_results)
    log.info('Num Unlabel {}'.format(num_unlabel_list[ii]))
    log.info('Test Acc = {:.3f} ({:.3f})'.format(_test_acc[0] * 100.0,
                                                 _test_acc[1] * 100.0))
  log.info('Experiment ID:')
  for ee, seed in zip(exp_ids, seed_list):
    print(ee, seed)


if __name__ == "__main__":
  main()
