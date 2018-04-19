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
import cv2
import numpy as np
import os
import gzip
import pickle as pkl
import tensorflow as tf

from fewshot.data.episode import Episode
from fewshot.data.data_factory import RegisterDataset
from fewshot.utils import logger

log = logger.get()

flags = tf.flags
flags.DEFINE_bool("disable_distractor", False,
                  "Whether or not to disable distractors")
flags.DEFINE_float("label_ratio", 0.1,
                   "Portion of labeled images in the training set.")
FLAGS = tf.flags.FLAGS


class MetaDataset(object):

  def next(self):
    """Get a new episode training."""
    pass


class RefinementMetaDataset(object):
  """A few-shot learning dataset with refinement (unlabeled) training. images.
  """

  def __init__(self, split, nway, nshot, num_unlabel, num_distractor, num_test,
               label_ratio, shuffle_episode, seed):
    """Creates a meta dataset.
    Args:
      folder: String. Path to the dataset.
      split: String.
      nway: Int. N way classification problem, default 5.
      nshot: Int. N-shot classification problem, default 1.
      num_unlabel: Int. Number of unlabeled examples per class, default 2.
      num_distractor: Int. Number of distractor classes, default 0.
      num_test: Int. Number of query images, default 10.
      split_def: String. "vinyals" or "lake", using different split definitions.
      aug_90: Bool. Whether to augment the training data by rotating 90 degrees.
      seed: Int. Random seed.
    """
    self._split = split
    self._nway = nway
    self._nshot = nshot
    self._num_unlabel = num_unlabel
    self._rnd = np.random.RandomState(seed)
    self._seed = seed
    self._num_distractor = 0 if FLAGS.disable_distractor else num_distractor
    log.warning("Number of distractors in each episode: {}".format(
        self._num_distractor))
    self._num_test = num_test
    self._label_ratio = FLAGS.label_ratio if label_ratio is None else label_ratio
    log.info('Label ratio {}'.format(self._label_ratio))
    self._shuffle_episode = shuffle_episode

    self.read_dataset()

    # Build a set for quick query.
    self._label_split_idx = np.array(self._label_split_idx)
    self._label_split_idx_set = set(list(self._label_split_idx))
    self._unlabel_split_idx = list(
        filter(lambda _idx: _idx not in self._label_split_idx_set,
               range(self._labels.shape[0])))
    self._unlabel_split_idx = np.array(self._unlabel_split_idx)
    if len(self._unlabel_split_idx) > 0:
      self._unlabel_split_idx_set = set(self._unlabel_split_idx)
    else:
      self._unlabel_split_idx_set = set()

    num_label_cls = len(self._label_str)
    self._num_classes = num_label_cls
    num_ex = self._labels.shape[0]
    ex_ids = np.arange(num_ex)
    self._label_idict = {}
    for cc in range(num_label_cls):
      self._label_idict[cc] = ex_ids[self._labels == cc]
    self._nshot = nshot

  def read_dataset(self):
    """Reads data from folder or cache."""
    raise NotImplemented()

  def label_split(self):
    """Gets label/unlabel image splits.
    Returns:
      labeled_split: List of int.
    """
    log.info('Label split using seed {:d}'.format(self._seed))
    rnd = np.random.RandomState(self._seed)
    num_label_cls = len(self._label_str)
    num_ex = self._labels.shape[0]
    ex_ids = np.arange(num_ex)

    labeled_split = []
    for cc in range(num_label_cls):
      cids = ex_ids[self._labels == cc]
      rnd.shuffle(cids)
      labeled_split.extend(cids[:int(len(cids) * self._label_ratio)])
    log.info("Total number of classes {}".format(num_label_cls))
    log.info("Labeled split {}".format(len(labeled_split)))
    log.info("Total image {}".format(num_ex))
    return sorted(labeled_split)

  def next(self, within_category=False, catcode=None):
    """Gets a new episode.
    within_category: bool. Whether or not to choose the N classes
    to all belong to the same more general category.
    (Only applicable for datasets with self._category_labels defined).

    within_category: bool. Whether or not to restrict the episode's classes
    to belong to the same general category (only applicable for JakeImageNet).
    If True, a random general category will be chosen, unless catcode is set.

    catcode: str. (e.g. 'n02795169') if catcode is provided (is not None),
    then the classes chosen for this episode will be restricted
    to be synsets belonging to the more general category with code catcode.
    """

    if within_category or not catcode is None:
      assert hasattr(self, "_category_labels")
      assert hasattr(self, "_category_label_str")
      if catcode is None:
        # Choose a category for this episode's classes
        cat_idx = np.random.randint(len(self._category_label_str))
        catcode = self._catcode_to_syncode.keys()[cat_idx]
      cat_synsets = self._catcode_to_syncode[catcode]
      cat_synsets_str = [self._syncode_to_str[code] for code in cat_synsets]
      allowable_inds = []
      for str in cat_synsets_str:
        allowable_inds.append(np.where(np.array(self._label_str) == str)[0])
      class_seq = np.array(allowable_inds).reshape((-1))
    else:
      num_label_cls = len(self._label_str)
      class_seq = np.arange(num_label_cls)

    self._rnd.shuffle(class_seq)

    train_img_ids = []
    train_labels = []
    test_img_ids = []
    test_labels = []

    train_unlabel_img_ids = []
    non_distractor = []

    train_labels_str = []
    test_labels_str = []

    is_training = self._split in ["train", "trainval"]
    assert is_training or self._split in ["val", "test"]

    for ii in range(self._nway + self._num_distractor):

      cc = class_seq[ii]
      # print(cc, ii < self._nway)
      _ids = self._label_idict[cc]

      # Split the image IDs into labeled and unlabeled.
      _label_ids = list(
          filter(lambda _id: _id in self._label_split_idx_set, _ids))
      _unlabel_ids = list(
          filter(lambda _id: _id not in self._label_split_idx_set, _ids))
      self._rnd.shuffle(_label_ids)
      self._rnd.shuffle(_unlabel_ids)

      # Add support set and query set (not for distractors).
      if ii < self._nway:
        train_img_ids.extend(_label_ids[:self._nshot])

        # Use the rest of the labeled image as queries, if num_test = -1.
        QUERY_SIZE_LARGE_ERR_MSG = (
            "Query + reference should be less than labeled examples." +
            "Num labeled {} Num test {} Num shot {}".format(
                len(_label_ids), self._num_test, self._nshot))
        assert self._nshot + self._num_test <= len(
            _label_ids), QUERY_SIZE_LARGE_ERR_MSG

        if self._num_test == -1:
          if is_training:
            num_test = len(_label_ids) - self._nshot
          else:
            num_test = len(_label_ids) - self._nshot - self._num_unlabel
        else:
          num_test = self._num_test
          if is_training:
            assert num_test <= len(_label_ids) - self._nshot
          else:
            assert num_test <= len(_label_ids) - self._num_unlabel - self._nshot

        test_img_ids.extend(_label_ids[self._nshot:self._nshot + num_test])
        train_labels.extend([ii] * self._nshot)
        train_labels_str.extend([self._label_str[cc]] * self._nshot)
        test_labels.extend([ii] * num_test)
        test_labels_str.extend([self._label_str[cc]] * num_test)
        non_distractor.extend([1] * self._num_unlabel)
      else:
        non_distractor.extend([0] * self._num_unlabel)

      # Add unlabeled images here.
      if is_training:
        # Use labeled, unlabeled split here for refinement.
        train_unlabel_img_ids.extend(_unlabel_ids[:self._num_unlabel])

      else:
        # Copy test set for refinement.
        # This will only work if the test procedure is rolled out in a sequence.
        train_unlabel_img_ids.extend(_label_ids[
            self._nshot + num_test:self._nshot + num_test + self._num_unlabel])

    train_img = self.get_images(train_img_ids) / 255.0
    train_unlabel_img = self.get_images(train_unlabel_img_ids) / 255.0
    test_img = self.get_images(test_img_ids) / 255.0
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)
    train_labels_str = np.array(train_labels_str)
    test_labels_str = np.array(test_labels_str)
    non_distractor = np.array(non_distractor)

    test_ids_set = set(test_img_ids)
    for _id in train_unlabel_img_ids:
      assert _id not in test_ids_set

    if self._shuffle_episode:
      # log.fatal('')
      # Shuffle the sequence order in an episode. Very important for RNN based
      # meta learners.
      train_idx = np.arange(train_img.shape[0])
      self._rnd.shuffle(train_idx)
      train_img = train_img[train_idx]
      train_labels = train_labels[train_idx]

      train_unlabel_idx = np.arange(train_unlabel_img.shape[0])
      self._rnd.shuffle(train_unlabel_idx)
      train_unlabel_img = train_unlabel_img[train_unlabel_idx]

      test_idx = np.arange(test_img.shape[0])
      self._rnd.shuffle(test_idx)
      test_img = test_img[test_idx]
      test_labels = test_labels[test_idx]

    return Episode(
        train_img,
        train_labels,
        test_img,
        test_labels,
        x_unlabel=train_unlabel_img,
        y_unlabel=non_distractor,
        y_train_str=train_labels_str,
        y_test_str=test_labels_str)

  def reset(self):
    self._rnd = np.random.RandomState(self._seed)

  def get_size(self):
    """Gets the size of the supervised portion."""
    return len(self._label_split_idx)

  def get_batch_idx(self, idx):
    """Gets a fully supervised training batch for classification.

    Returns: A tuple of
      x: Input image batch [N, H, W, C].
      y: Label class integer ID [N].
    """
    return self._images[self._label_split_idx[idx]], self._labels[
        self._label_split_idx[idx]]

  def get_batch_idx_test(self, idx):
    """Gets the test set (unlabeled set) for the fully supervised training."""

    return self._images[self._unlabel_split_idx[idx]], self._labels[
        self._unlabel_split_idx[idx]]

  @property
  def num_classes(self):
    return self._num_classes
