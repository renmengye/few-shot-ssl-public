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
"""Another KMeans based semisupervised model. Adds another class for distractors. The distractor
has a zero vector for the mean representation, and a learned radius to capture the remainders.

Author: Mengye Ren (mren@cs.toronto.edu)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.distractor_utils import eval_distractor
from fewshot.models.kmeans_refine_model import KMeansRefineModel
from fewshot.models.kmeans_utils import assign_cluster_radii
from fewshot.models.kmeans_utils import compute_logits_radii
from fewshot.models.kmeans_utils import update_cluster
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import concat
from fewshot.utils import logger

log = logger.get()

flags = tf.flags
flags.DEFINE_bool("learn_radius", True,
                  "Whether or not to learn distractor radius.")
flags.DEFINE_float("init_radius", 100.0, "Initial radius for the distractors.")
FLAGS = tf.flags.FLAGS


@RegisterModel("kmeans-refine-radius")
class KMeansRefineRadiusModel(KMeansRefineModel):

  def predict(self):
    """See `model.py` for documentation."""
    nclasses = self.nway
    num_cluster_steps = self.config.num_cluster_steps
    h_train, h_unlabel, h_test = self.get_encoded_inputs(
        self.x_train, self.x_unlabel, self.x_test)
    y_train = self.y_train
    protos = self._compute_protos(nclasses, h_train, y_train)

    # Distractor class has a zero vector as prototype.
    protos = concat([protos, tf.zeros_like(protos[:, 0:1, :])], 1)

    # Hard assignment for training images.
    prob_train = [None] * (nclasses + 1)
    for kk in range(nclasses):
      # [B, N, 1]
      prob_train[kk] = tf.expand_dims(
          tf.cast(tf.equal(y_train, kk), h_train.dtype), 2)
      prob_train[-1] = tf.zeros_like(prob_train[0])
    prob_train = concat(prob_train, 2)

    # Initialize cluster radii.
    radii = [None] * (nclasses + 1)
    y_train_shape = tf.shape(y_train)
    bsize = y_train_shape[0]
    for kk in range(nclasses):
      radii[kk] = tf.ones([bsize, 1]) * 1.0

    # Distractor class has a larger radius.
    if FLAGS.learn_radius:
      log_distractor_radius = tf.get_variable(
          "log_distractor_radius",
          shape=[],
          dtype=tf.float32,
          initializer=tf.constant_initializer(np.log(FLAGS.init_radius)))
      distractor_radius = tf.exp(log_distractor_radius)
    else:
      distractor_radius = FLAGS.init_radius
    distractor_radius = tf.cond(
        tf.shape(self._x_unlabel)[1] > 0, lambda: distractor_radius,
        lambda: 100000.0)
    # distractor_radius = tf.Print(distractor_radius, [distractor_radius])
    radii[-1] = tf.ones([bsize, 1]) * distractor_radius
    radii = concat(radii, 1)  # [B, K]

    h_all = concat([h_train, h_unlabel], 1)
    logits_list = []
    logits_list.append(compute_logits_radii(protos, h_test, radii))

    # Run clustering.
    for tt in range(num_cluster_steps):
      # Label assignment.
      prob_unlabel = assign_cluster_radii(protos, h_unlabel, radii)
      prob_all = concat([prob_train, prob_unlabel], 1)
      protos = update_cluster(h_all, prob_all)
      logits_list.append(compute_logits_radii(protos, h_test, radii))

    # Distractor evaluation.
    is_distractor = tf.equal(tf.argmax(prob_unlabel, axis=-1), nclasses)
    pred_non_distractor = 1.0 - tf.to_float(is_distractor)
    acc, recall, precision = eval_distractor(pred_non_distractor,
                                             self.y_unlabel)
    self._non_distractor_acc = acc
    self._distractor_recall = recall
    self._distractor_precision = precision
    self._distractor_pred = 1.0 - tf.exp(prob_unlabel[:, :, -1])
    return logits_list
