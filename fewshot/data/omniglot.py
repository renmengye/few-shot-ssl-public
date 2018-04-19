# Copyright (c) 2018 Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell,
# Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle, Richars S. Zemel.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import cv2
import numpy as np
import os
import pickle as pkl

import tensorflow as tf

from fewshot.data.episode import Episode
from fewshot.data.data_factory import RegisterDataset
from fewshot.data.refinement_dataset import RefinementMetaDataset
from fewshot.utils import logger

log = logger.get()
flags = tf.flags
FLAGS = tf.flags.FLAGS


def get_image_folder(folder, split_def, split):
  if split_def == 'lake':
    if split == 'train':
      folder_ = os.path.join(folder, 'images_background')
    else:
      folder_ = os.path.join(folder, 'images_evaluation')
  elif split_def == 'vinyals':
    folder_ = os.path.join(folder, 'images_all')
  return folder_


def get_vinyals_split_file(split):
  curdir = os.path.dirname(os.path.realpath(__file__))
  split_file = os.path.join(curdir, 'omniglot_split', '{}.txt'.format(split))
  return split_file


def read_lake_split(folder, aug_90=False):
  """Reads dataset from folder."""
  subfolders = os.listdir(folder)
  label_idx = []
  label_str = []
  data = []
  for sf in subfolders:
    sf_ = os.path.join(folder, sf)
    img_fnames = os.listdir(sf_)
    for character in img_fnames:
      char_folder = os.path.join(sf_, character)
      img_list = os.listdir(char_folder)
      for img_fname in img_list:
        fname_ = os.path.join(char_folder, img_fname)
        img = cv2.imread(fname_)
        # Shrink images.
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.minimum(255, np.maximum(0, img))
        img = 255 - img[:, :, 0:1]
        if aug_90:
          M = cv2.getRotationMatrix2D((14, 14), 90, 1)
          dst = img
          for ii in range(4):
            dst = cv2.warpAffine(dst, M, (28, 28))
            data.append(np.expand_dims(np.expand_dims(dst, 0), 3))
            label_idx.append(len(label_str) + ii)
        else:
          img = np.expand_dims(img, 0)
          data.append(img)
          label_idx.append(len(label_str))

      if aug_90:
        for ii in range(4):
          label_str.append(sf + '_' + character + '_' + str(ii))
      else:
        label_str.append(sf + '_' + character)
  print('Number of classes {}'.format(len(label_str)))
  print('Number of images {}'.format(len(data)))
  images = np.concatenate(data, axis=0)
  labels = np.array(label_idx, dtype=np.int32)
  label_str = label_str
  return images, labels, label_str


def read_vinyals_split(folder, split_file, aug_90=False):
  """Reads dataset from a folder with a split file."""
  lines = open(split_file, 'r').readlines()
  lines = map(lambda x: x.strip('\n\r'), lines)
  label_idx = []
  label_str = []
  data = []
  for ff in lines:
    char_folder = os.path.join(folder, ff)
    img_list = os.listdir(char_folder)
    for img_fname in img_list:
      fname_ = os.path.join(char_folder, img_fname)
      img = cv2.imread(fname_)
      img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
      img = np.minimum(255, np.maximum(0, img))
      img = 255 - img[:, :, 0:1]
      if aug_90:
        M = cv2.getRotationMatrix2D((14, 14), 90, 1)
        dst = img
        for ii in range(4):
          dst = cv2.warpAffine(dst, M, (28, 28))
          data.append(np.expand_dims(np.expand_dims(dst, 0), 3))
          label_idx.append(len(label_str) + ii)
      else:
        img = np.expand_dims(img, 0)
        data.append(img)
        label_idx.append(len(label_str))
    if aug_90:
      for ii in range(4):
        label_str.append(ff + '_' + str(ii))
    else:
      label_str.append(ff)
  print('Number of classes {}'.format(len(label_str)))
  print('Number of images {}'.format(len(data)))
  images = np.concatenate(data, axis=0)
  labels = np.array(label_idx, dtype=np.int32)
  return images, labels, label_str


@RegisterDataset('omniglot')
class OmniglotDataset(RefinementMetaDataset):
  """A few-shot learning dataset with refinement (unlabeled) training. images.
  """

  def __init__(self,
               folder,
               split,
               nway=5,
               nshot=1,
               num_unlabel=5,
               num_distractor=5,
               num_test=-1,
               split_def='vinyals',
               label_ratio=None,
               aug_90=True,
               shuffle_episode=False,
               seed=0):
    """Creates a meta dataset.
    Args:
      folder: String. Path to the Omniglot dataset.
      split: String. 'train' or 'test' for Lake's split, 'train', 'trainval',
        'val', test' for Vinyals' split.
      nway: Int. N way classification problem, default 5.
      nshot: Int. N-shot classification problem, default 1.
      num_unlabel: Int. Number of unlabeled examples per class, default 2.
      num_distractor: Int. Number of distractor classes, default 0.
      num_test: Int. Number of query images, default 10.
      split_def: String. 'vinyals' or 'lake', using different split definitions.
      aug_90: Bool. Whether to augment the training data by rotating 90 degrees.
      seed: Int. Random seed.
    """
    self._folder = folder
    self._aug_90 = aug_90
    self._split_def = split_def
    self._split = split
    if FLAGS.disable_distractor:
      num_distractor = 0
    super(OmniglotDataset,
          self).__init__(split, nway, nshot, num_unlabel, num_distractor,
                         num_test, label_ratio, shuffle_episode, seed)

  def get_images(self, inds):
    return self._images[inds]

  def read_cache(self):
    """Reads dataset from cached pklz file."""
    cache_path = self.get_cache_path()
    print(cache_path)
    if os.path.exists(cache_path):
      try:
        with open(cache_path, 'rb') as f:
          data = pkl.load(f, encoding='bytes')
          self._images = data[b'images']
          self._labels = data[b'labels']
          self._label_str = data[b'label_str']
      except:
        with open(cache_path, 'rb') as f:
          data = pkl.load(f)
          self._images = data['images']
          self._labels = data['labels']
          self._label_str = data['label_str']
      self.read_label_split()
      return True
    else:
      return False

  def read_label_split(self):
    cache_path_labelsplit = self.get_label_split_path()
    if os.path.exists(cache_path_labelsplit):
      self._label_split_idx = np.loadtxt(cache_path_labelsplit, dtype=np.int64)
    else:
      if self._split in ['train', 'trainval']:
        log.info('Use {}% image for labeled split.'.format(
            int(self._label_ratio * 100)))
        self._label_split_idx = self.label_split()
      elif self._split in ['val', 'test']:
        log.info('Use all image in labeled split, since we are in val/test')
        self._label_split_idx = np.arange(self._images.shape[0])
      else:
        raise ValueError('Unknown split {}'.format(self._split))
      self._label_split_idx = np.array(self.label_split(), dtype=np.int64)
      self.save_label_split()

  def save_cache(self):
    """Saves pklz cache."""
    data = {
        'images': self._images,
        'labels': self._labels,
        'label_str': self._label_str,
    }
    with open(self.get_cache_path(), 'wb') as f:
      pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

  def save_label_split(self):
    np.savetxt(self.get_label_split_path(), self._label_split_idx, fmt='%d')

  def read_dataset(self):
    # Read data from folder or cache.
    if not self.read_cache():
      folder, split_def, split = self._folder, self._split_def, self._split
      folder = get_image_folder(folder, split_def, split)
      if split_def == 'lake':
        self._images, self._labels, self._label_str = read_lake_split(
            folder, aug_90=self._aug_90)
      elif split_def == 'vinyals':
        split_file = get_vinyals_split_file(self._split)
        self._images, self._labels, self._label_str = read_vinyals_split(
            folder, split_file, aug_90=self._aug_90)
      self.read_label_split()
      self.save_cache()

  def get_label_split_path(self):
    aug_str = '_aug90' if self._aug_90 else ''
    split_def_str = '_' + self._split_def
    label_ratio_str = '_' + str(int(self._label_ratio * 100))
    seed_id_str = '_' + str(self._seed)
    if self._split in ['train', 'trainval']:
      cache_path = os.path.join(
          self._folder, self._split + split_def_str + aug_str + '_labelsplit' +
          label_ratio_str + seed_id_str + '.txt')
    elif self._split in ['val', 'test']:
      cache_path = os.path.join(
          self._folder,
          self._split + split_def_str + aug_str + '_labelsplit' + '.txt')
    return cache_path

  def get_cache_path(self):
    """Gets cache file name."""
    aug_str = '_aug90' if self._aug_90 else ''
    split_def_str = '_' + self._split_def
    cache_path = os.path.join(self._folder,
                              self._split + split_def_str + aug_str + '.pkl')

    return cache_path
