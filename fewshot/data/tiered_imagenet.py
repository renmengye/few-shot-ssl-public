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
"""Dataset API for tieredImageNet

Author: Eleni Triantafillou (eleni@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import cv2
import csv
import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf

from tqdm import tqdm

from fewshot.data.data_factory import RegisterDataset
from fewshot.data.episode import Episode
from fewshot.data.refinement_dataset import RefinementMetaDataset
from fewshot.data.compress_tiered_imagenet import decompress
from fewshot.utils import logger

log = logger.get()

flags = tf.flags
flags.DEFINE_bool(
    "load_images", True,
    "Whether to store the images themselves or their paths when loading the dataset."
)
FLAGS = tf.flags.FLAGS


@RegisterDataset("tiered-imagenet")
class TieredImageNetDataset(RefinementMetaDataset):
  """A few-shot learning dataset with refinement (unlabeled) training. images.
  """

  def __init__(self,
               folder,
               split,
               nway=5,
               nshot=1,
               num_unlabel=5,
               num_distractor=5,
               num_test=5,
               label_ratio=None,
               shuffle_episode=False,
               seed=0,
               aug_90=False):
    """Creates a meta dataset.
    Args:
      folder: String. Path to the Omniglot dataset.
      split: String. "train" or "test" for Lake's split, "train", "trainval",
        "val", test" for Vinyals' split.
      nway: Int. N way classification problem, default 5.
      nshot: Int. N-shot classification problem, default 1.
      num_unlabel: Int. Number of unlabeled examples per class, default 2.
      num_distractor: Int. Number of distractor classes, default 0.
      num_test: Int. Number of query images, default 10.
      aug_90: Bool. Whether to augment the training data by rotating 90 degrees.
      seed: Int. Random seed.
      use_specific_labels: bool. Whether to use specific or general labels.
    """
    self._folder = folder
    self._data_folder = folder
    self._imagenet_train_folder = os.path.join(self._data_folder, "train")
    self._splits_folder = self.get_splits_folder()
    print("num unlabel {}".format(num_unlabel))
    print("num test {}".format(num_test))
    print("num distractor {}".format(num_distractor
                                     if not FLAGS.disable_distractor else 0))
    super(TieredImageNetDataset,
          self).__init__(split, nway, nshot, num_unlabel, num_distractor,
                         num_test, label_ratio, shuffle_episode, seed)

    # Dictionary mapping categories to their synsets
    self._catcode_to_syncode = self.build_catcode_to_syncode()
    self._catcode_to_str = self.build_catcode_to_str()
    self._syncode_to_str = self.build_syncode_to_str()

    # Inverse dictionaries.
    num_ex = self._label_specific.shape[0]
    ex_ids = np.arange(num_ex)
    num_label_cls_specific = len(self._label_specific_str)
    self._label_specific_idict = {}
    for cc in range(num_label_cls_specific):
      self._label_specific_idict[cc] = ex_ids[self._label_specific == cc]

  def get_splits_folder(self):
    curdir = os.path.dirname(os.path.realpath(__file__))
    split_dir = os.path.join(curdir, "tiered_imagenet_split")
    if not os.path.exists(split_dir):
      raise ValueError("split_dir {} does not exist.".format(split_dir))
    return split_dir

  def get_label_split_path(self):
    label_ratio_str = '_' + str(int(self._label_ratio * 100))
    seed_id_str = '_' + str(self._seed)
    if self._split in ['train', 'trainval']:
      cache_path = os.path.join(
          self._folder,
          self._split + '_labelsplit' + label_ratio_str + seed_id_str + '.txt')
    elif self._split in ['val', 'test']:
      cache_path = os.path.join(self._folder,
                                self._split + '_labelsplit' + '.txt')
    return cache_path

  def get_cache_path(self):
    """Gets cache file name."""
    cache_path = os.path.join(self._folder, self._split)
    cache_path_labels = cache_path + "_labels.pkl"
    cache_path_images = cache_path + "_images.npz"
    return cache_path_labels, cache_path_images

  def read_cache(self):
    """Reads dataset from cached pkl file."""
    cache_path_labels, cache_path_images = self.get_cache_path()

    # Decompress images.
    if not os.path.exists(cache_path_images):
      png_pkl = cache_path_images[:-4] + '_png.pkl'
      if os.path.exists(png_pkl):
        decompress(cache_path_images, png_pkl)
      else:
        return False
    if os.path.exists(cache_path_labels) and os.path.exists(cache_path_images):
      print("Read cached labels from {}".format(cache_path_labels))
      try:
        with open(cache_path_labels, "rb") as f:
          data = pkl.load(f, encoding='bytes')
          self._label_specific = data[b"label_specific"]
          self._label_general = data[b"label_general"]
          self._label_specific_str = data[b"label_specific_str"]
          self._label_general_str = data[b"label_general_str"]
      except:
        with open(cache_path_labels, "rb") as f:
          data = pkl.load(f)
          self._label_specific = data["label_specific"]
          self._label_general = data["label_general"]
          self._label_specific_str = data["label_specific_str"]
          self._label_general_str = data["label_general_str"]
        self._label_str = self._label_specific_str
        self._labels = self._label_specific
      print("Read cached images from {}".format(cache_path_images))
      with np.load(cache_path_images, mmap_mode="r", encoding='latin1') as data:
        self._images = data["images"]
        print("self._images.shape {}".format(self._images.shape))
      self.read_label_split()
      return True
    else:
      return False

  def save_cache(self):
    """Saves pkl cache."""

    cache_path_labels, cache_path_images = self.get_cache_path(self._seed)
    data = {
        "label_specific": self._label_specific,
        "label_general": self._label_general,
        "label_specific_str": self._label_specific_str,
        "label_general_str": self._label_general_str,
    }
    if not os.path.exists(cache_path_labels):
      with open(cache_path_labels, "wb") as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)
      print("Saved cache in location {}".format(self.get_cache_path()[0]))
    # Save the images
    if not os.path.exists(cache_path_images):
      np.savez(self.get_cache_path()[1], images=self._images)
      print("Saved the images in location {}".format(self.get_cache_path()[1]))

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

  def save_label_split(self):
    np.savetxt(self.get_label_split_path(), self._label_split_idx, fmt='%d')

  def read_splits(self):
    """
    Returns a list of labels belonging to the given split
    (as specified by self._split).
    Each element of this list is a (specific_label, general_label)
    tuple.
    :return:
    """
    specific_label, general_label = [], []
    csv_path = os.path.join(self._splits_folder, self._split + '.csv')
    with open(csv_path, 'r') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for i, row in enumerate(csvreader):
        # Sometimes there's an empty row at the bottom
        if len(row) == 0:
          break
        specific_label.append(row[0])
        general_label.append(row[1])

    num_specific_classes = list(set(specific_label))
    num_general_classes = list(set(general_label))
    print(
        'Found {} synsets belonging to a total of {} categories for split {}.'.
        format(
            len(num_specific_classes), len(num_general_classes), self._split))
    return specific_label, general_label

  def read_dataset(self):
    if not self.read_cache():
      specific_classes, general_classes = self.read_splits()
      label_idx_specific = []
      label_idx_general = []
      label_str_specific = []
      label_str_general = []
      data = []
      synset_dirs = os.listdir(self._imagenet_train_folder)
      for synset in tqdm(synset_dirs, desc="Reading dataset..."):
        if not synset in specific_classes:
          continue
        for cat, synset_list in self._catcode_to_syncode.iteritems():
          if synset in synset_list:
            break
        synset_dir_path = os.path.join(self._imagenet_train_folder, synset)
        img_list = os.listdir(synset_dir_path)
        for img_fname in img_list:
          fpath = os.path.join(synset_dir_path, img_fname)
          if FLAGS.load_images:
            img = cv2.imread(fpath)
            img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_CUBIC)
            img = np.expand_dims(img, 0)
            data.append(img)
          else:
            data_paths.append(fpath)
          label_idx_specific.append(len(label_str_specific))
          label_idx_general.append(len(label_str_general))

        synset_name = self._syncode_to_str[synset]
        category_name = self._catcode_to_str[cat]
        label_str_specific.append(synset_name)
        if category_name not in label_str_general:
          label_str_general.append(category_name)
      print("Number of synsets {}".format(len(label_str_specific)))
      print("label_str_general {}".format(label_str_general))
      print("len label_str_general {}".format(len(label_str_general)))
      labels_specific = np.array(label_idx_specific, dtype=np.int32)
      labels_general = np.array(label_idx_general, dtype=np.int32)
      images = np.concatenate(data, axis=0)
      self._images = images
      self._label_specific = labels_specific
      self._label_general = labels_general
      self._label_specific_str = label_str_specific
      self._label_general_str = label_str_general
      self.read_label_split()
      self.save_cache()

  def build_catcode_to_syncode(self):
    catcode_to_syncode = {}
    csv_path = os.path.join(self._splits_folder, self._split + '.csv')
    print(csv_path)
    with open(csv_path, 'r') as csvfile:
      csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
      for i, row in enumerate(csvreader):
        # Sometimes there's an empty row at the bottom
        if len(row) == 0:
          break

        if not row[1] in catcode_to_syncode:
          # Adding synset label row[0] to the list synsets belonging to category row[1]
          catcode_to_syncode[row[1]] = []
        if not row[0] in catcode_to_syncode[row[1]]:
          catcode_to_syncode[row[1]].append(row[0])
    print(
        "Created mapping from category to their synset codes with {} entries.".
        format(len(catcode_to_syncode)))
    return catcode_to_syncode

  def build_syncode_to_str(self):
    """
    Build a mapping from synsets to the (string)
    description of the corresponding class.
    :return:
    """
    path_str = os.path.join(self._data_folder, "class_names.txt")
    path_synsets = os.path.join(self._data_folder, "synsets.txt")
    with open(path_str, "r") as f:
      lines_str = f.readlines()
    with open(path_synsets, "r") as f:
      lines_synsets = f.readlines()
    syn_to_str = {}
    for l_str, l_syn in zip(lines_str, lines_synsets):
      syn_to_str[l_syn.strip()] = l_str.strip()
    return syn_to_str

  def build_catcode_to_str(self):
    synset2words = {}
    path = os.path.join(self._splits_folder, "words.txt")
    for _, row in pd.read_fwf(
        path, header=None, names=['synset', 'words'], usecols=[0,
                                                               1]).iterrows():
      synset2words[row.synset] = row.words
    return synset2words

  def get_images(self, inds=None):
    imgs = self._images[inds]
    return imgs
