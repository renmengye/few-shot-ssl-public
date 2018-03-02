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
from __future__ import print_function, division

import tensorflow as tf


def eval_distractor(pred_non_distractor, gt_non_distractor):
  """Evaluates distractor prediction.

  Args:
    pred_non_distractor
    gt_non_distractor

  Returns:
    acc:
    recall:
    precision:
  """
  y = gt_non_distractor
  pred_distractor = 1.0 - pred_non_distractor
  non_distractor_correct = tf.to_float(tf.equal(pred_non_distractor, y))
  distractor_tp = pred_distractor * (1 - y)
  distractor_recall = tf.reduce_sum(distractor_tp) / tf.reduce_sum(1 - y)
  distractor_precision = tf.reduce_sum(distractor_tp) / (
      tf.reduce_sum(pred_distractor) +
      tf.to_float(tf.equal(tf.reduce_sum(pred_distractor), 0.0)))
  acc = tf.reduce_mean(non_distractor_correct)
  recall = tf.reduce_mean(distractor_recall)
  precision = tf.reduce_mean(distractor_precision)

  return acc, recall, precision
