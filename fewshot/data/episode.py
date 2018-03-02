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
class Episode(object):

  def __init__(self,
               x_train,
               y_train,
               x_test,
               y_test,
               x_unlabel=None,
               y_unlabel=None,
               y_train_str=None,
               y_test_str=None):
    """Creates a miniImageNet episode.
    Args:
      x_train:  [N, ...]. Training data.
      y_train: [N]. Training label.
      x_test: [N, ...]. Testing data.
      y_test: [N]. Testing label.
    """
    self._x_train = x_train
    self._y_train = y_train
    self._x_test = x_test
    self._y_test = y_test
    self._x_unlabel = x_unlabel
    self._y_unlabel = y_unlabel
    self._y_train_str = y_train_str
    self._y_test_str = y_test_str

  def next_batch(self):
    return self

  @property
  def x_train(self):
    return self._x_train

  @property
  def x_test(self):
    return self._x_test

  @property
  def y_train(self):
    return self._y_train

  @property
  def y_test(self):
    return self._y_test

  @property
  def x_unlabel(self):
    return self._x_unlabel

  @property
  def y_unlabel(self):
    return self._y_unlabel

  @property
  def y_train_str(self):
    return self._y_train_str

  @property
  def y_test_str(self):
    return self._y_test_str
