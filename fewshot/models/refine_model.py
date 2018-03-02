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
"""
A few-shot classification model implementation that refines on unlabled
refinement images.

Author: Mengye Ren (mren@cs.toronto.edu)

A single episode is divided into three parts:
  1) Labeled reference images (self.x_ref).
  2) Unlabeled refinement images (self.x_unlabel).
  3) Labeled query images (from validation) (self.x_candidate).
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.nnlib import cnn, weight_variable, concat
from fewshot.models.basic_model import BasicModel
from fewshot.utils import logger
log = logger.get()

# Load up the LSTM cell implementation.
if tf.__version__.startswith("0"):
  BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell
  LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
else:
  BasicLSTMCell = tf.contrib.rnn.BasicLSTMCell
  LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple


class RefineModel(BasicModel):
  """A retrieval model with an additional refinement stage."""

  def __init__(self,
               config,
               nway=1,
               nshot=1,
               num_unlabel=10,
               candidate_size=10,
               is_training=True,
               dtype=tf.float32):
    """Initiliazer.
    Args:
      config: Model configuration object.
      nway: Int. Number of classes in the reference images.
      nshot: Int. Number of labeled reference images.
      num_unlabel: Int. Number of unlabeled refinement images.
      candidate_size: Int. Number of candidates in the query stage.
      is_training: Bool. Whether is in training mode.
      dtype: TensorFlow data type.
    """
    self._num_unlabel = num_unlabel
    self._x_unlabel = tf.placeholder(
        dtype, [None, None, config.height, config.width, config.num_channel],
        name="x_unlabel")
    self._y_unlabel = tf.placeholder(dtype, [None, None], name="y_unlabel")
    super(RefineModel, self).__init__(
        config,
        nway=nway,
        nshot=nshot,
        num_test=candidate_size,
        is_training=is_training,
        dtype=dtype)

  @property
  def x_unlabel(self):
    return self._x_unlabel

  @property
  def y_unlabel(self):
    return self._y_unlabel


if __name__ == "__main__":
  from fewshot.configs.omniglot_config import OmniglotRefineConfig
  model = RefineModel(OmniglotRefineConfig())
