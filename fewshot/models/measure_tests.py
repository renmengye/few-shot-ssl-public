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

import numpy as np
import unittest

from fewshot.models.measure import batch_apk, apk


def fake_batch_apk(logits, pos_mask, k):
  ap = []
  for ii in range(logits.shape[0]):
    ap.append(apk(logits[ii], pos_mask[ii], k[ii]))
  return np.array(ap)


class MeasureTests(unittest.TestCase):

  def test_batch_apk(self):
    rnd = np.random.RandomState(0)
    for ii in range(100):
      logits = rnd.uniform(0.0, 1.0, [10, 12])
      pos_mask = (rnd.uniform(0.0, 1.0, [10, 12]) > 0.5).astype(np.float32)
      k = rnd.uniform(5.0, 10.0, [10]).astype(np.int32)
      ap1 = batch_apk(logits, pos_mask, k)
      ap2 = fake_batch_apk(logits, pos_mask, k)
      np.testing.assert_allclose(ap1, ap2)


if __name__ == "__main__":
  unittest.main()
