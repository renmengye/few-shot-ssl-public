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
import cv2
import numpy as np
import six
import sys
import pickle as pkl

from tqdm import tqdm


def compress(path, output):
  with np.load(path, mmap_mode="r") as data:
    images = data["images"]
    array = []
    for ii in tqdm(six.moves.xrange(images.shape[0]), desc='compress'):
      im = images[ii]
      im_str = cv2.imencode('.png', im)[1]
      array.append(im_str)
  with open(output, 'wb') as f:
    pkl.dump(array, f, protocol=pkl.HIGHEST_PROTOCOL)


def decompress(path, output):
  with open(output, 'rb') as f:
    array = pkl.load(f)
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  for ii, item in tqdm(enumerate(array), desc='decompress'):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)


def main():
  if sys.argv[1] == 'compress':
    compress(sys.argv[2], sys.argv[3])
  elif sys.argv[1] == 'decompress':
    decompress(sys.argv[2], sys.argv[3])


if __name__ == '__main__':
  main()
