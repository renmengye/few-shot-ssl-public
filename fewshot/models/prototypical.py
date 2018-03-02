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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from fewshot.models.nnlib import cnn
from fewshot.models.nnlib import concat
from fewshot.utils import logger

import tensorflow as tf

log = logger.get()


def debug(x, idx):
  return tf.Print(x,
                  [idx, tf.reduce_mean(x), tf.reduce_max(x), tf.reduce_min(x)])


def compute_logits(cluster_centers, data):
  """Computes the logits of being in one cluster, squared Euclidean.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    data: [N, D] Data representation.
  Returns:
    log_prob: [N, K] logits.
  """
  cluster_centers = tf.expand_dims(cluster_centers, 0)  # [1, K, D]
  data = tf.expand_dims(data, 1)  # [N, 1, D]
  neg_dist = -tf.reduce_sum(tf.square(data - cluster_centers), [2])
  return neg_dist


def compute_gmm_diag_logits(cluster_centers, cluster_covar, data, nclasses):
  """Computes the logits of being in one cluster, diagonal Manhalanobis.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    cluster_covar: [K, D] Cluster covariance matrix.
    data: [N, D] Data representation.
    nclasses: Integer. K, number of classes.
  Returns:
    log_prob: [N, K] logits.
  """
  cluster_centers = tf.expand_dims(cluster_centers, 0)  # [1, K, D]
  data = tf.expand_dims(data, 1)  # [N, 1, D]
  diff = data - cluster_centers  # [N, K, D]
  cluster_covar = tf.minimum(tf.maximum(cluster_covar, 0.1), 5.0)
  cluster_covar_inv = 1.0 / cluster_covar
  return tf.reduce_sum(-cluster_covar_inv * diff * diff, [2])


def compute_gmm_logits(cluster_centers, cluster_covar, data, nclasses):
  """Computes the logits of being in one cluster, Manhalanobis.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    cluster_covar: [K, D, D] Cluster covariance matrix.
    data: [N, D] Data representation.
    nclasses: Integer. K, number of classes.
  Returns:
    log_prob: [N, K] logits.
  """
  cluster_centers = tf.expand_dims(cluster_centers, 0)  # [1, K, D]
  data = tf.expand_dims(data, 1)  # [N, 1 D]
  diff = data - cluster_centers  # [N, K, D]
  result = []  # [N, K]
  print("covar", cluster_covar.get_shape())
  for kk in range(nclasses):
    _covar = cluster_covar[kk, :, :]  # [D, D]
    _diff = diff[:, kk, :]  # [N, D]
    _diff_ = tf.expand_dims(_diff, 2)
    _icovar = tf.matrix_inverse(_covar)
    _icovar = tf.expand_dims(_icovar, 0)  # [1, D, D]
    prod = tf.reduce_sum(_diff_ * _icovar, [1])  # [N, D]
    print(_icovar.get_shape())
    print(_diff_.get_shape())
    print(prod.get_shape())
    prod = tf.reduce_sum(_diff * prod, [1], keep_dims=True)  # [N, 1]
    result.append(-prod)
  logits = concat(result, 1)
  print("logits", logits.get_shape())
  return logits


def sq_dist_loss(cluster_centers, data):
  """Squared distance based loss function.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    data: [N, D] Data representation.
  Returns:
    loss: Average of all minimum distance towards cluster center.
  """
  min_dist = tf.reduce_min(-compute_logits(cluster_centers, data), [1])
  return tf.reduce_mean(min_dist)


def mh_dist_loss(cluster_centers, cluster_covar, data, nclasses):
  """Mahalanobis distance loss.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    cluster_covar: [K, D, D] Cluster covariance matrix.
    data: [N, D] Data representation.
    nclasses: Integer. K, number of classes.
  Returns:
    loss: Average of all minimum distance towards cluster center.
  """
  min_dist = tf.reduce_min(
      -compute_gmm_logits(cluster_centers, cluster_covar, data, nclasses), [1])
  return tf.reduce_mean(min_dist)


def mh_diag_dist_loss(cluster_centers, cluster_covar, data, nclasses):
  """Mahalanobis diagonal distance loss.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    cluster_covar: [K, D] Cluster covariance matrix.
    data: [N, D] Data representation.
    nclasses: Integer. K, number of classes.
  Returns:
    loss: Average of all minimum distance towards cluster center.
  """
  min_dist = tf.reduce_min(
      -compute_gmm_diag_logits(cluster_centers, cluster_covar, data, nclasses),
      [1])
  return tf.reduce_mean(min_dist)


def assign_cluster(cluster_centers, data):
  """Assigns data to cluster center, using K-Means.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    data: [N, D] Data representation.
  Returns:
    prob: [N, K] Soft assignment.
  """
  logits = compute_logits(cluster_centers, data)
  prob = tf.nn.softmax(logits)
  return prob


def assign_gmm_diag_cluster(cluster_centers, cluster_covar, data, nclasses):
  """Assigns data to cluster center, using GMM.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    cluster_covar: [K, D, D] Covariance matrix for each cluster.
    data: [N, D] Data representation.
    nclasses: Integer. K, number of classes.
  Returns:
    prob: [N, K] Soft assignment.
  """
  logits = compute_gmm_diag_logits(cluster_centers, cluster_covar, data,
                                   nclasses)
  prob = tf.nn.softmax(logits)
  return prob


def assign_gmm_cluster(cluster_centers, cluster_covar, data, nclasses):
  """Assigns data to cluster center, using GMM.
  Args:
    cluster_centers: [K, D] Cluster center representation.
    cluster_covar: [K, D, D] Covariance matrix for each cluster.
    data: [N, D] Data representation.
    nclasses: Integer. K, number of classes.
  Returns:
    prob: [N, K] Soft assignment.
  """
  logits = compute_gmm_logits(cluster_centers, cluster_covar, data, nclasses)
  prob = tf.nn.softmax(logits)
  return prob


def update_cluster(data, prob):
  """Updates cluster center based on assignment, standard K-Means.
  Args:
    data: [N, D]. Data representation.
    prob: [N, K]. Cluster assignment soft probability.
  Returns:
    cluster_centers: [K, D]. Cluster center representation.
  """
  # Normalize accross N.
  prob2 = prob / tf.reduce_sum(prob, [0], keep_dims=True)
  return tf.reduce_sum(tf.expand_dims(data, 1) * tf.expand_dims(prob2, 2), [0])


def update_gmm_diag_cluster(data, prob):
  """Updates cluster center based on assignment, GMM.
  Args:
    data: [N, D] Data representation.
    prob: [N, K] Cluster assignment soft probability.
  Returns:
    cluster_centers: [K, D].
  """
  prob2 = prob / tf.reduce_sum(prob, [0], keep_dims=True)  # [N, K]
  d1 = tf.expand_dims(data, 1)  # [N, 1, D]
  var = d1 * d1
  prob2 = tf.expand_dims(prob2, 2)  # [N, K, 1]
  d1_mean = tf.reduce_sum(d1 * prob2, [0], keep_dims=True)  # [1, K, D]
  #d1_mean = tf.Print(d1_mean, [d1_mean[0]])
  d1_ = d1 - d1_mean  # [N, K, D]
  #d1_ = tf.Print(d1_, [tf.shape(d1_)])
  var = d1_ * d1_
  var = tf.reduce_sum(var * prob2, [0])  # [K, D]
  mean = tf.squeeze(d1_mean, 0)  # [K, D]
  #var = debug(var, 10)
  #mean = debug(mean, 11)
  return mean, var


def update_gmm_cluster(data, prob):
  """Updates cluster center based on assignment, GMM.
  Args:
    data: [N, D] Data representation.
    prob: [N, K] Cluster assignment soft probability.
  Returns:
    cluster_centers: [K, D].
  """
  print("prob", prob.get_shape())
  prob2 = prob / tf.reduce_sum(prob, [0], keep_dims=True)  # [N, K]
  mean = tf.reduce_sum(tf.expand_dims(data, 1) * tf.expand_dims(prob2, 2), [0])
  d1 = tf.expand_dims(data, 1)  # [N, 1, D]
  # [N, 1, 1, D] * [N, 1, D, 1]
  var = tf.expand_dims(d1, 2) * tf.expand_dims(d1, 3)  # [N, 1, D, D]
  print("haha", var.get_shape())
  prob2 = tf.expand_dims(tf.expand_dims(prob2, 2), 3)  # [N, K, 1, 1]
  print("kk", prob2.get_shape())
  # [N, 1, D, D] * [N, K, 1, 1] = [K, D, D]
  var = tf.reduce_sum(var * prob2, [0])
  print("udpate covar", var.get_shape())
  return mean, var


def prototypical_layer(nclasses, x_train, y_train, x_test, phi, ext_wts=None):
  """Computes the prototypes, cluster centers.
  Args:
    x_train: [N, ...], Train data.
    y_train: [N], Train class labels.
    x_test: [N, ...], Test data.
    phi: Feature extractor function.
  Returns:
    logits: [N, K], Test prediction.
  """
  protos = [None] * nclasses
  x_all = concat([x_train, x_test], 0)
  h, _ = phi(x_all, reuse=None, is_training=True, ext_wts=ext_wts)
  num_x_train = tf.shape(x_train)[0]
  h_train = h[:num_x_train, :]
  h_test = h[num_x_train:, :]
  for kk in range(nclasses):
    ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype),
                          1)  # [N, 1]
    protos[kk] = tf.reduce_sum(h_train * ksel, [0], keep_dims=True)  # [N, D]
    protos[kk] /= tf.reduce_sum(ksel)
    protos[kk] = tf.Print(protos[kk], [
        'proto', tf.reduce_mean(protos[kk]), tf.reduce_max(protos[kk]),
        tf.reduce_min(protos[kk])
    ])
  protos = concat(protos, 0)  # [K, D]
  logits = compute_logits(protos, h_test)
  return logits


def prototypical_clustering_layer(nclasses,
                                  x_train,
                                  y_train,
                                  x_test,
                                  phi,
                                  num_cluster_steps,
                                  lambd=0.1,
                                  alpha=0.0):
  """Computes the prototypes, cluster centers, with additional clustering on
    the validation data.
  Args:
    x_train: [N, D], Train data.
    y_train: [N], Train class labels.
    x_test: [N, D], Test data.
    phi: Feature extractor function.
  Returns:
    logits: [N, K], Test prediction.
  """
  protos = [None] * nclasses
  x_all = concat([x_train, x_test], 0)
  h, _ = phi(x_all, reuse=None, is_training=True)
  num_x_train = tf.shape(x_train)[0]
  h_train = h[:num_x_train, :]
  h_test = h[num_x_train:, :]

  # Initialize cluster center.
  for kk in range(nclasses):
    ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype),
                          1)  # [N, 1]
    protos[kk] = tf.reduce_sum(h_train * ksel, [0], keep_dims=True)  # [N, D]
    protos[kk] /= tf.reduce_sum(ksel)
  protos = concat(protos, 0)  # [K, D]

  # Run clustering.
  for tt in range(num_cluster_steps):
    all_data = concat([h_train, h_test], 0)
    # Label assignment.
    prob = assign_cluster(protos, all_data)
    # Prototype update.
    ## This is a vanilla version.
    ## Probably want to impose a constraint that each training example can only
    ## be in one cluster.
    protos = update_cluster(all_data, prob)

  # Be cautious here!!
  # Returns the logits and unsupervised clustering loss.
  uloss = sq_dist_loss(protos, all_data)
  logits = compute_logits(protos, h_test)
  return logits, uloss


def prototypical_clustering_gmm_layer(nclasses,
                                      x_train,
                                      y_train,
                                      x_test,
                                      phi,
                                      num_cluster_steps,
                                      lambd=0.1,
                                      alpha=0.0):
  """Computes the prototypes, cluster centers, with additional clustering on
    the validation data.
  Args:
    x_train: [N, D], Train data.
    y_train: [N], Train class labels.
    x_test: [N, D], Test data.
    phi: Feature extractor function.
  Returns:
    logits: [N, K], Test prediction.
  """
  protos = [None] * nclasses
  covar = [None] * nclasses
  x_all = concat([x_train, x_test], 0)
  h, _ = phi(x_all, reuse=None, is_training=True)
  num_x_train = tf.shape(x_train)[0]
  h_train = h[:num_x_train, :]
  h_test = h[num_x_train:, :]
  ndim = tf.shape(h)[1]

  # Initialize cluster center.
  for kk in range(nclasses):
    ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype),
                          1)  # [N, 1]
    protos[kk] = tf.reduce_sum(h_train * ksel, [0], keep_dims=True)  # [N, D]
    protos[kk] /= tf.reduce_sum(ksel)
    #covar[kk] = tf.expand_dims(tf.eye(ndim), 0)
    covar[kk] = tf.ones([1, ndim])  # diagonal
  protos = concat(protos, 0)  # [K, D]
  covar = concat(covar, 0)  # [K, D, D]

  # Run clustering.
  for tt in range(num_cluster_steps):
    all_data = concat([h_train, h_test], 0)
    # Label assignment.
    prob = assign_gmm_diag_cluster(protos, covar, all_data, nclasses)
    protos, covar = update_gmm_diag_cluster(all_data, prob)

  # Be cautious here!!
  uloss = mh_diag_dist_loss(protos, covar, all_data, nclasses)
  logits = compute_gmm_diag_logits(protos, covar, h_test, nclasses)
  return logits, uloss


def prototypical_clustering_learn_layer(nclasses,
                                        x_train,
                                        y_train,
                                        x_test,
                                        phi,
                                        num_cluster_steps,
                                        lambd=0.1,
                                        alpha=0.0):
  """Computes the prototypes, cluster centers, with additional clustering on
    the validation data.
  Args:
    x_train: [N, D], Train data.
    y_train: [N], Train class labels.
    x_test: [N, D], Test data.
    phi: Feature extractor function.
  Returns:
    logits: [N, K], Test prediction.
  """
  protos = [None] * nclasses
  x_all = concat([x_train, x_test], 0)
  h, wts = phi(x_all, reuse=None, is_training=True)
  num_x_train = tf.shape(x_train)[0]
  h_train = h[:num_x_train, :]
  h_test = h[num_x_train:, :]
  wts_keys = wts.keys()
  wts_tensors = [wts[kk] for kk in wts_keys]

  for kk in range(nclasses):
    ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype),
                          1)  # [N, 1]
    protos[kk] = tf.reduce_sum(h_train * ksel, [0], keep_dims=True)  # [N, D]
    protos[kk] /= tf.reduce_sum(ksel)

  protos = concat(protos, 0)  # [K, D]

  for tt in range(num_cluster_steps):
    # Compute new representation of the data.
    h, _ = phi(
        x_all,
        reuse=True,
        is_training=True,
        ext_wts=dict(zip(wts_keys, wts_tensors)))
    h_train = h[:num_x_train, :]
    h_test = h[num_x_train:, :]
    all_data = concat([h_train, h_test], 0)

    # Label assignment.
    prob = assign_cluster(protos, all_data)

    # Prototype update.
    ## This is a vanilla version.
    ## Probably want to impose a constraint that each training example can only
    ## be in one cluster.
    protos = update_cluster(all_data, prob)

    if alpha > 0.0 and tt < num_cluster_steps - 1:
      # We can also use soft labels here.
      loss = lambd * sq_dist_loss(protos, all_data)
      # One gradient update towards fast weights.
      print(wts_tensors)
      [print(vv.name) for vv in wts_tensors]
      grads = tf.gradients(loss, wts_tensors, gate_gradients=True)
      [print(gg) for gg in grads]
      # Stop the gradient of the gradient.
      grads = [tf.stop_gradient(gg) for gg in grads]
      wts_tensors = [wt - alpha * gg for wt, gg in zip(wts_tensors, grads)]

  # Be cautious here!!
  #uloss = sq_dist_loss(protos, all_data)
  uloss = 0.0
  logits = compute_logits(protos, h_test)
  return logits, uloss


def prototypical_model(config,
                       nclasses,
                       is_training,
                       x_train,
                       y_train,
                       x_test,
                       dtype=tf.float32,
                       ext_wts=None):
  """Builds a simple prototypical network.
  Args:
    config: Configuration object.
    nclasses: Number of classes.
    x_train: [N, ...]. Train input.
    y_train: [N]. Training labels.
    x_test: [N, ...]. Test input.
  Returns:
    y_test: [N, K]. Test prediction.
  """

  def phi(x, reuse=None, scope="Model", is_training=True, ext_wts=None):
    """Feature extraction function.

    Args:
      x: [N, H, W, C]. Input.
      reuse: Whether to reuse variables here.
      scope: Variable scope.
      ext_wts: A dictionary of external weights to be used from.
    """
    with tf.name_scope(scope):
      with tf.variable_scope("Model", reuse=reuse):
        h, wts = cnn(
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
            ext_wts=ext_wts)
        h_size = reduce(lambda x, y: x * y,
                        [int(ss) for ss in h.get_shape()[1:]])
        h = tf.reshape(h, [-1, h_size])
        return h, wts

  if not hasattr(config, 'prototype_clustering'):
    config.prototype_clustering = False
  if config.prototype_clustering:
    if config.fast_learn_rate == 0.0:
      if config.clustering_method == "kmeans":
        _layerfn = prototypical_clustering_layer
      elif config.clustering_method == "gmm":
        _layerfn = prototypical_clustering_gmm_layer
      logits, loss = _layerfn(
          nclasses,
          x_train,
          y_train,
          x_test,
          phi,
          config.num_cluster_steps,
          lambd=config.clustering_loss_lambda)
    else:
      logits, loss = prototypical_clustering_learn_layer(
          nclasses,
          x_train,
          y_train,
          x_test,
          phi,
          config.num_cluster_steps,
          lambd=config.clustering_loss_lambda,
          alpha=config.fast_learn_rate)
    if config.clustering_loss_lambda > 0.0:
      loss *= config.clustering_loss_lambda
  else:
    logits = prototypical_layer(
        nclasses, x_train, y_train, x_test, phi, ext_wts=ext_wts)
    loss = None
  return logits, loss
