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
from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from fewshot.models.nnlib import concat, round_st
from fewshot.utils import logger

log = logger.get()


def compute_logits(cluster_centers, data):
  """Computes the logits of being in one cluster, squared Euclidean.
  Args:
    cluster_centers: [B, K, D] Cluster center representation.
    data: [B, N, D] Data representation.
  Returns:
    log_prob: [B, N, K] logits.
  """
  cluster_centers = tf.expand_dims(cluster_centers, 1)  # [B, 1, K, D]
  data = tf.expand_dims(data, 2)  # [B, N, 1, D]
  # [B, N, K]
  neg_dist = -tf.reduce_sum(tf.square(data - cluster_centers), [-1])
  return neg_dist


def assign_cluster(cluster_centers, data):
  """Assigns data to cluster center, using K-Means.
  Args:
    cluster_centers: [B, K, D] Cluster center representation.
    data: [B, N, D] Data representation.
  Returns:
    prob: [B, N, K] Soft assignment.
  """
  logits = compute_logits(cluster_centers, data)
  logits_shape = tf.shape(logits)
  bsize = logits_shape[0]
  ndata = logits_shape[1]
  ncluster = logits_shape[2]
  logits = tf.reshape(logits, [-1, ncluster])
  prob = tf.nn.softmax(logits)  # Use softmax distance.
  prob = tf.reshape(prob, [bsize, ndata, ncluster])
  return prob


def update_cluster(data, prob, fix_last_row=False):
  """Updates cluster center based on assignment, standard K-Means.
  Args:
    data: [B, N, D]. Data representation.
    prob: [B, N, K]. Cluster assignment soft probability.
    fix_last_row: Bool. Whether or not to fix the last row to 0.
  Returns:
    cluster_centers: [B, K, D]. Cluster center representation.
  """
  # Normalize accross N.
  if fix_last_row:
    prob_ = prob[:, :, :-1]
  else:
    prob_ = prob
  prob_sum = tf.reduce_sum(prob_, [1], keep_dims=True)
  prob_sum += tf.to_float(tf.equal(prob_sum, 0.0))
  prob2 = prob_ / prob_sum
  cluster_centers = tf.reduce_sum(
      tf.expand_dims(data, 2) * tf.expand_dims(prob2, 3), [1])
  if fix_last_row:
    cluster_centers = concat(
        [cluster_centers,
         tf.zeros_like(cluster_centers[:, 0:1, :])], 1)
  return cluster_centers


def assign_cluster_radii(cluster_centers, data, radii):
  """Assigns data to cluster center, using K-Means.

  Args:
    cluster_centers: [B, K, D] Cluster center representation.
    data: [B, N, D] Data representation.
    radii: [B, K] Cluster radii.
  Returns:
    prob: [B, N, K] Soft assignment.
  """
  logits = compute_logits_radii(cluster_centers, data, radii)
  logits_shape = tf.shape(logits)
  bsize = logits_shape[0]
  ndata = logits_shape[1]
  ncluster = logits_shape[2]
  logits = tf.reshape(logits, [-1, ncluster])
  prob = tf.nn.softmax(logits)
  prob = tf.reshape(prob, [bsize, ndata, ncluster])
  return prob


def compute_logits_radii(cluster_centers, data, radii):
  """Computes the logits of being in one cluster, squared Euclidean.

  Args:
    cluster_centers: [B, K, D] Cluster center representation.
    data: [B, N, D] Data representation.
    radii: [B, K] Cluster radii.
  Returns:
    log_prob: [B, N, K] logits.
  """
  cluster_centers = tf.expand_dims(cluster_centers, 1)  # [B, 1, K, D]
  data = tf.expand_dims(data, 2)  # [B, N, 1, D]
  radii = tf.expand_dims(radii, 1)  # [B, 1, K]
  # [B, N, K]
  neg_dist = -tf.reduce_sum(tf.square(data - cluster_centers), [-1])
  logits = neg_dist / 2.0 / (radii**2)
  norm_constant = 0.5 * tf.log(2 * np.pi) + tf.log(radii)
  logits -= norm_constant
  return logits


def assign_cluster_soft_mask(cluster_centers, data, mask):
  """Assigns data to cluster center, using K-Means.
  Args:
    cluster_centers: [B, K, D] Cluster center representation.
    data: [B, N, D] Data representation.
    mask: [B, N, K] Mask for each cluster.
  Returns:
    prob: [B, N, K] Soft assignment.
  """
  logits = compute_logits(cluster_centers, data)
  logits_shape = tf.shape(logits)
  bsize = logits_shape[0]
  ndata = logits_shape[1]
  ncluster = logits_shape[2]
  logits = tf.reshape(logits, [-1, ncluster])
  prob = tf.nn.softmax(logits)  # Use softmax distance.
  prob = tf.reshape(prob, [bsize, ndata, ncluster]) * mask
  return prob, mask
