from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import sys

is_py2 = sys.version[0] == "2"
if is_py2:
  import Queue as queue
else:
  import queue as queue
import threading

from fewshot.data.batch_iter import IBatchIterator, BatchIterator
from fewshot.utils import logger


class BatchProducer(threading.Thread):

  def __init__(self, q, batch_iter):
    super(BatchProducer, self).__init__()
    threading.Thread.__init__(self)
    self.q = q
    self.batch_iter = batch_iter
    self.log = logger.get()
    self._stoper = threading.Event()
    self.daemon = True

  def stop(self):
    self._stoper.set()

  def stopped(self):
    return self._stoper.isSet()

  def run(self):
    while not self.stopped():
      try:
        b = self.batch_iter.next()
        self.q.put(b)
      except StopIteration:
        self.q.put(None)
        break


class BatchConsumer(threading.Thread):

  def __init__(self, q):
    super(BatchConsumer, self).__init__()
    self.q = q
    self.daemon = True
    self._stoper = threading.Event()

  def stop(self):
    self._stoper.set()

  def stopped(self):
    return self._stoper.isSet()

  def run(self):
    while not self.stopped():
      try:
        self.q.get(False)
        self.q.task_done()
      except queue.Empty:
        pass


class ConcurrentBatchIterator(IBatchIterator):

  def __init__(self,
               batch_iter,
               max_queue_size=10,
               num_threads=5,
               log_queue=20,
               name=None):
    """
        Data provider wrapper that supports concurrent data fetching.
        """
    super(ConcurrentBatchIterator, self).__init__()
    self.max_queue_size = max_queue_size
    self.num_threads = num_threads
    self.q = queue.Queue(maxsize=max_queue_size)
    self.log = logger.get()
    self.batch_iter = batch_iter
    self.fetchers = []
    self.init_fetchers()
    self.counter = 0
    self.relaunch = True
    self._stopped = False
    self.log_queue = log_queue
    self.name = name

  def __len__(self):
    return len(self.batch_iter)

  def init_fetchers(self):
    for ii in six.moves.xrange(self.num_threads):
      f = BatchProducer(self.q, self.batch_iter)
      f.start()
      self.fetchers.append(f)

  def get_name(self):
    if self.name is not None:
      return "Queue \"{}\":".format(self.name)
    else:
      return ""

  def info(self, message):
    self.log.info("{} {}".format(self.get_name(), message), verbose=2)

  def warning(self, message):
    self.log.warning("{} {}".format(self.get_name(), message))

  def scan(self, do_print=False):
    dead = []
    num_alive = 0
    for ff in self.fetchers:
      if not ff.is_alive():
        dead.append(ff)
        self.info("Found one dead thread.")
        if self.relaunch:
          self.info("Relaunch")
          fnew = BatchProducer(self.q, self.batch_iter)
          fnew.start()
          self.fetchers.append(fnew)
      else:
        num_alive += 1
    if do_print:
      self.info("Number of alive threads: {}".format(num_alive))
      s = self.q.qsize()
      if s > self.max_queue_size / 3:
        self.info("Data queue size: {}".format(s))
      else:
        self.warning("Data queue size: {}".format(s))
    for dd in dead:
      self.fetchers.remove(dd)

  def next(self):
    if self._stopped:
      raise StopIteration
    self.scan(do_print=(self.counter % self.log_queue == 0))
    if self.counter % self.log_queue == 0:
      self.counter = 0
    batch = self.q.get()
    self.q.task_done()
    self.counter += 1
    while batch is None:
      self.info("Got an empty batch. Ending iteration.")
      self.relaunch = False
      try:
        batch = self.q.get(False)
        self.q.task_done()
        qempty = False
      except queue.Empty:
        qempty = True
        pass

      if qempty:
        self.info("Queue empty. Scanning for alive thread.")
        # Scan for alive thread.
        found_alive = False
        for ff in self.fetchers:
          if ff.is_alive():
            found_alive = True
            break

        self.info("No alive thread found. Joining.")
        # If no alive thread, join all.
        if not found_alive:
          for ff in self.fetchers:
            ff.join()
          self._stopped = True
          raise StopIteration
      else:
        self.info("Got another batch from the queue.")
    return batch

  def reset(self):
    self.info("Resetting concurrent batch iter")
    self.info("Stopping all workers")
    for f in self.fetchers:
      f.stop()
    self.info("Cleaning queue")
    cleaner = BatchConsumer(self.q)
    cleaner.start()
    for f in self.fetchers:
      f.join()
    self.q.join()
    cleaner.stop()
    self.info("Resetting index")
    self.batch_iter.reset()
    self.info("Restarting workers")
    self.fetchers = []
    self.init_fetchers()
    self.relaunch = True
    self._stopped = False


if __name__ == "__main__":
  from batch_iter import BatchIterator
  b = BatchIterator(100, batch_size=6, get_fn=None)
  cb = ConcurrentBatchIterator(b, max_queue_size=5, num_threads=3)
  for _batch in cb:
    log = logger.get()
    log.info(("Final out", _batch))
