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
A prototypical network with K-means to refine unlabeled examples.

Author: Mengye Ren (mren@cs.toronto.edu)

In a single episode, the model computes the mean representation of the
positive refereene images as prototypes and then refine the representation by
running a few steps of soft K-means iterations.
"""
import tensorflow as tf

from fewshot.models.kmeans_utils import assign_cluster, update_cluster, compute_logits
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import concat
from fewshot.models.refine_model import RefineModel
from fewshot.utils import logger

log = logger.get()


@RegisterModel("kmeans-refine")
class KMeansRefineModel(RefineModel):

  def predict(self):
    """See `model.py` for documentation."""
    nclasses = self.nway
    num_cluster_steps = self.config.num_cluster_steps
    h_train, h_unlabel, h_test = self.get_encoded_inputs(
        self.x_train, self.x_unlabel, self.x_test)
    y_train = self.y_train
    protos = self._compute_protos(nclasses, h_train, y_train)
    logits = compute_logits(protos, h_test)

    # Hard assignment for training images.
    prob_train = [None] * nclasses
    for kk in range(nclasses):
      # [B, N, 1]
      prob_train[kk] = tf.expand_dims(
          tf.cast(tf.equal(y_train, kk), h_train.dtype), 2)
    prob_train = concat(prob_train, 2)

    h_all = concat([h_train, h_unlabel], 1)

    logits_list = []
    logits_list.append(compute_logits(protos, h_test))

    # Run clustering.
    for tt in range(num_cluster_steps):
      # Label assignment.
      prob_unlabel = assign_cluster(protos, h_unlabel)
      entropy = tf.reduce_sum(
          -prob_unlabel * tf.log(prob_unlabel), [2], keep_dims=True)
      prob_all = concat([prob_train, prob_unlabel], 1)
      prob_all = tf.stop_gradient(prob_all)
      protos = update_cluster(h_all, prob_all)
      # protos = tf.cond(
      #     tf.shape(self._x_unlabel)[1] > 0,
      #     lambda: update_cluster(h_all, prob_all), lambda: protos)
      logits_list.append(compute_logits(protos, h_test))

    return logits_list
