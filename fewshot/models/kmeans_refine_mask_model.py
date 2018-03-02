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
""" Another KMeans based semisupervised model. Predict a mask based on the neighbor distance
distribution.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.distractor_utils import eval_distractor
from fewshot.models.kmeans_refine_model import KMeansRefineModel
from fewshot.models.kmeans_utils import assign_cluster_soft_mask
from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.kmeans_utils import update_cluster
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import concat, mlp
from fewshot.utils import logger

log = logger.get()

flags = tf.flags
FLAGS = tf.flags.FLAGS


@RegisterModel("kmeans-refine-mask")
class KMeansRefineMaskModel(KMeansRefineModel):

  def predict(self):
    """See `model.py` for documentation."""
    nclasses = self.nway
    num_cluster_steps = self.config.num_cluster_steps
    h_train, h_unlabel, h_test = self.get_encoded_inputs(
        self.x_train, self.x_unlabel, self.x_test)
    y_train = self.y_train
    protos = self._compute_protos(nclasses, h_train, y_train)
    logits_list = []
    logits_list.append(compute_logits(protos, h_test))

    # Hard assignment for training images.
    prob_train = [None] * (nclasses)
    for kk in range(nclasses):
      # [B, N, 1]
      prob_train[kk] = tf.expand_dims(
          tf.cast(tf.equal(y_train, kk), h_train.dtype), 2)
    prob_train = concat(prob_train, 2)

    y_train_shape = tf.shape(y_train)
    bsize = y_train_shape[0]

    h_all = concat([h_train, h_unlabel], 1)
    mask = None

    # Calculate pairwise distances.
    protos_1 = tf.expand_dims(protos, 2)
    protos_2 = tf.expand_dims(h_unlabel, 1)
    pair_dist = tf.reduce_sum((protos_1 - protos_2)**2, [3])  # [B, K, N]
    mean_dist = tf.reduce_mean(pair_dist, [2], keep_dims=True)
    pair_dist_normalize = pair_dist / mean_dist
    min_dist = tf.reduce_min(
        pair_dist_normalize, [2], keep_dims=True)  # [B, K, 1]
    max_dist = tf.reduce_max(pair_dist_normalize, [2], keep_dims=True)
    mean_dist, var_dist = tf.nn.moments(
        pair_dist_normalize, [2], keep_dims=True)
    mean_dist += tf.to_float(tf.equal(mean_dist, 0.0))
    var_dist += tf.to_float(tf.equal(var_dist, 0.0))
    skew = tf.reduce_mean(
        ((pair_dist_normalize - mean_dist)**3) / (tf.sqrt(var_dist)**3), [2],
        keep_dims=True)
    kurt = tf.reduce_mean(
        ((pair_dist_normalize - mean_dist)**4) / (var_dist**2) - 3, [2],
        keep_dims=True)

    n_features = 5
    n_out = 3

    dist_features = tf.reshape(
        concat([min_dist, max_dist, var_dist, skew, kurt], 2),
        [-1, n_features])  # [BK, 4]
    dist_features = tf.stop_gradient(dist_features)

    hdim = [n_features, 20, n_out]
    act_fn = [tf.nn.tanh, None]
    thresh = mlp(
        dist_features,
        hdim,
        is_training=True,
        act_fn=act_fn,
        dtype=tf.float32,
        add_bias=True,
        wd=None,
        init_std=[0.01, 0.01],
        init_method=None,
        scope="dist_mlp",
        dropout=None,
        trainable=True)
    scale = tf.exp(thresh[:, 2])
    bias_start = tf.exp(thresh[:, 0])
    bias_add = thresh[:, 1]
    bias_start = tf.reshape(bias_start, [bsize, 1, -1])  #[B, 1, K]
    bias_add = tf.reshape(bias_add, [bsize, 1, -1])

    self._scale = scale
    self._bias_start = bias_start
    self._bias_add = bias_add

    # Run clustering.
    for tt in range(num_cluster_steps):
      protos_1 = tf.expand_dims(protos, 2)
      protos_2 = tf.expand_dims(h_unlabel, 1)
      pair_dist = tf.reduce_sum((protos_1 - protos_2)**2, [3])  # [B, K, N]
      m_dist = tf.reduce_mean(pair_dist, [2])  # [B, K]
      m_dist_1 = tf.expand_dims(m_dist, 1)  # [B, 1, K]
      m_dist_1 += tf.to_float(tf.equal(m_dist_1, 0.0))
      # Label assignment.
      if num_cluster_steps > 1:
        bias_tt = bias_start + (tt / float(num_cluster_steps - 1)) * bias_add
      else:
        bias_tt = bias_start

      negdist = compute_logits(protos, h_unlabel)
      mask = tf.sigmoid((negdist / m_dist_1 + bias_tt) * scale)
      prob_unlabel, mask = assign_cluster_soft_mask(protos, h_unlabel, mask)
      prob_all = concat([prob_train, prob_unlabel * mask], 1)
      # No update if 0 unlabel.
      protos = tf.cond(
          tf.shape(self._x_unlabel)[1] > 0,
          lambda: update_cluster(h_all, prob_all), lambda: protos)
      logits_list.append(compute_logits(protos, h_test))

    # Distractor evaluation.
    if mask is not None:
      max_mask = tf.reduce_max(mask, [2])
      mean_mask = tf.reduce_mean(max_mask)
      pred_non_distractor = tf.to_float(max_mask > mean_mask)
      acc, recall, precision = eval_distractor(pred_non_distractor,
                                               self.y_unlabel)
      self._non_distractor_acc = acc
      self._distractor_recall = recall
      self._distractor_precision = precision
      self._distractor_pred = max_mask
    return logits_list
