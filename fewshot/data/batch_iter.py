"""
A batch iterator.
Usage:
    for idx in BatchIterator(num=1000, batch_size=25):
        inp_batch = inp_all[idx]
        labels_batch = labels_all[idx]
        train(inp_batch, labels_batch)
"""
from __future__ import division, print_function

import numpy as np
import threading

from fewshot.utils.logger import get as get_logger


class IBatchIterator(object):

  def __iter__(self):
    """Get iterable."""
    return self

  def next(self):
    raise Exception("Not implemented")

  def reset(self):
    raise Exception("Not implemented")


class BatchIterator(IBatchIterator):

  def __init__(self,
               num,
               batch_size=1,
               progress_bar=False,
               log_epoch=10,
               get_fn=None,
               cycle=False,
               shuffle=True,
               stagnant=False,
               seed=2,
               num_batches=-1):
    """Construct a batch iterator.
        Args:
            data: numpy.ndarray, (N, D), N is the number of examples, D is the
            feature dimension.
            labels: numpy.ndarray, (N), N is the number of examples.
            batch_size: int, batch size.
        """

    self._num = num
    self._batch_size = batch_size
    self._step = 0
    self._num_steps = int(np.ceil(self._num / float(batch_size)))
    if num_batches > 0:
      self._num_steps = min(self._num_steps, num_batches)
    self._pb = None
    self._variables = None
    self._get_fn = get_fn
    self.get_fn = get_fn
    self._cycle = cycle
    self._shuffle_idx = np.arange(self._num)
    self._shuffle = shuffle
    self._random = np.random.RandomState(seed)
    if shuffle:
      self._random.shuffle(self._shuffle_idx)
    self._shuffle_flag = False
    self._stagnant = stagnant
    self._log_epoch = log_epoch
    self._log = get_logger()
    self._epoch = 0
    if progress_bar:
      self._pb = pb.get(self._num_steps)
      pass
    self._mutex = threading.Lock()
    pass

  def __iter__(self):
    """Get iterable."""
    return self

  def __len__(self):
    """Get iterable length."""
    return self._num_steps

  @property
  def variables(self):
    return self._variables

  def set_variables(self, variables):
    self._variables = variables

    def get_fn(idx):
      return self._get_fn(idx, variables=variables)

    self.get_fn = get_fn
    return self

  def reset(self):
    self._step = 0

  def print_progress(self):
    e = self._epoch
    a = (self._step * self._batch_size) % self._num
    b = self._num
    p = a / b * 100
    digit = int(np.ceil(np.log10(b)))
    progress_str = "{:" + str(digit) + "d}"
    progress_str = (progress_str + "/" + progress_str).format(int(a), int(b))
    self._log.info("Epoch {:3d} Progress {} ({:5.2f}%)".format(
        e, progress_str, p))
    pass

  def next(self):
    """Iterate next element."""
    self._mutex.acquire()
    try:
      # Shuffle data.
      if self._shuffle_flag:
        self._random.shuffle(self._shuffle_idx)
        self._shuffle_flag = False

      # Read/write of self._step stay in a thread-safe block.
      if not self._cycle:
        if self._step >= self._num_steps:
          raise StopIteration()

      # Calc start/end based on current step.
      start = self._batch_size * self._step
      end = self._batch_size * (self._step + 1)

      # Progress bar.
      if self._pb is not None:
        self._pb.increment()

      # Epoch record.
      if self._cycle:
        if int(end / self._num) > int(start / self._num):
          self._epoch += 1

      # Increment step.
      if not self._stagnant:
        self._step += 1

      # Print progress
      if self._log_epoch > 0 and self._step % self._log_epoch == 0:
        self.print_progress()
    finally:
      self._mutex.release()

    if not self._cycle:
      end = min(self._num, end)
      idx = np.arange(start, end)
      idx = idx.astype("int")
      if self.get_fn is not None:
        return self.get_fn(idx)
      else:
        return idx
    else:
      start = start % self._num
      end = end % self._num
      if end > start:
        idx = np.arange(start, end)
        idx = idx.astype("int")
        idx = self._shuffle_idx[idx]
      else:
        idx = np.array(range(start, self._num) + range(0, end))
        idx = idx.astype("int")
        idx = self._shuffle_idx[idx]
        # Shuffle every cycle.
        if self._shuffle:
          self._shuffle_flag = True
      if self.get_fn is not None:
        return self.get_fn(idx)
      else:
        return idx


if __name__ == "__main__":
  b = BatchIterator(
      400,
      batch_size=32,
      progress_bar=False,
      get_fn=lambda x: x,
      cycle=False,
      shuffle=False)
  for ii in b:
    print(ii)
  b.reset()
  for ii in b:
    print(ii)
