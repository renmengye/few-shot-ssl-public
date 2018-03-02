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
"""Learning rate scheduler utilities."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from fewshot.utils import logger

log = logger.get()


class FixedLearnRateScheduler(object):
  """Adjusts learning rate according to a fixed schedule."""

  def __init__(self, sess, model, base_lr, lr_decay_steps, lr_list=None):
    """
    Args:
      sess: TensorFlow session object.
      model: Model object.
      base_lr: Base learning rate.
      lr_decay_steps: A list of step number which we perform learning decay.
      lr_list: A list of learning rate decay multiplier. By default, all 0.1.
    """
    self.model = model
    self.sess = sess
    self.lr = base_lr
    self.lr_list = lr_list
    self.lr_decay_steps = lr_decay_steps
    self.model.assign_lr(self.sess, self.lr)

  def step(self, niter):
    """Adds to counter. Adjusts learning rate if necessary.

    Args:
      niter: Current number of iterations.
    """
    if len(self.lr_decay_steps) > 0:
      if (niter + 1) == self.lr_decay_steps[0]:
        if self.lr_list is not None:
          self.lr = self.lr_list[0]
        else:
          self.lr *= 0.1  ## Divide 10 by default!!!
        self.model.assign_lr(self.sess, self.lr)
        self.lr_decay_steps.pop(0)
        log.warning("LR decay steps {}".format(self.lr_decay_steps))
        if self.lr_list is not None:
          self.lr_list.pop(0)
      elif (niter + 1) > self.lr_decay_steps[0]:
        ls = self.lr_decay_steps
        while len(ls) > 0 and (niter + 1) > ls[0]:
          ls.pop(0)
          log.warning("LR decay steps {}".format(self.lr_decay_steps))
          if self.lr_list is not None:
            self.lr = self.lr_list.pop(0)
          else:
            self.lr *= 0.1
        self.model.assign_lr(self.sess, self.lr)


class ExponentialLearnRateScheduler(object):
  """Adjusts learning rate according to an exponential decay schedule."""

  def __init__(self, sess, model, base_lr, offset_steps, total_steps, final_lr,
               interval):
    """
    Args:
      sess: TensorFlow session object.
      model: Model object.
      base_lr: Base learning rate.
      offset_steps: Initial non-decay steps.
      total_steps: Total number of steps.
      final_lr: Final learning rate by the end of training.
      interval: Number of steps in between learning rate updates (staircase).
    """
    self.model = model
    self.sess = sess
    self.lr = base_lr
    self.offset_steps = offset_steps
    self.total_steps = total_steps
    self.time_constant = (total_steps - offset_steps) / np.log(
        base_lr / final_lr)
    self.final_lr = final_lr
    self.interval = interval
    self.model.assign_lr(self.sess, self.lr)

  def step(self, niter):
    """Adds to counter. Adjusts learning rate if necessary.

    Args:
      niter: Current number of iterations.
    """
    if niter > self.offset_steps:
      steps2 = niter - self.offset_steps
      if steps2 % self.interval == 0:
        new_lr = base_lr * np.exp(-steps2 / self.time_constant)
        self.model.assign_lr(self.sess, new_lr)
