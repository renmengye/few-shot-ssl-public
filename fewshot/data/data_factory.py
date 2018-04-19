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
import os
import tensorflow as tf

from fewshot.data.concurrent_batch_iter import ConcurrentBatchIterator

flags = tf.flags
flags.DEFINE_string("data_root", "data", "Data root")
FLAGS = tf.flags.FLAGS

DATASET_REGISTRY = {}


def RegisterDataset(dataset_name):
  """Registers a dataset class"""

  def decorator(f):
    DATASET_REGISTRY[dataset_name] = f
    return f

  return decorator


def get_data_folder(dataset_name):
  data_folder = os.path.join(FLAGS.data_root, dataset_name)
  return data_folder


def get_dataset(dataset_name, split, *args, **kwargs):
  if dataset_name in DATASET_REGISTRY:
    return DATASET_REGISTRY[dataset_name](get_data_folder(dataset_name), split,
                                          *args, **kwargs)
  else:
    raise ValueError("Unknown dataset \"{}\"".format(dataset_name))


def get_concurrent_iterator(dataset, max_queue_size=100, num_threads=10):
  return ConcurrentBatchIterator(
      dataset,
      max_queue_size=max_queue_size,
      num_threads=num_threads,
      log_queue=-1)
