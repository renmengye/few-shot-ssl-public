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

import datetime
import os
import sys

from fewshot.utils import logger

log = logger.get()


class ExperimentLogger():
  """Writes experimental logs to CSV file."""

  def __init__(self, logs_folder):
    """Initialize files."""
    self._write_to_csv = logs_folder is not None

    if self._write_to_csv:
      if not os.path.isdir(logs_folder):
        os.makedirs(logs_folder)

      catalog_file = os.path.join(logs_folder, "catalog")

      with open(catalog_file, "w") as f:
        f.write("filename,type,name\n")

      with open(catalog_file, "a") as f:
        f.write("{},plain,{}\n".format("cmd.txt", "Commands"))

      with open(os.path.join(logs_folder, "cmd.txt"), "w") as f:
        f.write(" ".join(sys.argv))

      with open(catalog_file, "a") as f:
        f.write("train_ce.csv,csv,Train Loss (Cross Entropy)\n")
        f.write("train_acc.csv,csv,Train Accuracy\n")
        f.write("valid_acc.csv,csv,Validation Accuracy\n")
        f.write("learn_rate.csv,csv,Learning Rate\n")

      self.train_file_name = os.path.join(logs_folder, "train_ce.csv")
      if not os.path.exists(self.train_file_name):
        with open(self.train_file_name, "w") as f:
          f.write("step,time,ce\n")

      self.trainval_file_name = os.path.join(logs_folder, "train_acc.csv")
      if not os.path.exists(self.trainval_file_name):
        with open(self.trainval_file_name, "w") as f:
          f.write("step,time,acc\n")

      self.val_file_name = os.path.join(logs_folder, "valid_acc.csv")
      if not os.path.exists(self.val_file_name):
        with open(self.val_file_name, "w") as f:
          f.write("step,time,acc\n")

      self.lr_file_name = os.path.join(logs_folder, "learn_rate.csv")
      if not os.path.exists(self.lr_file_name):
        with open(self.lr_file_name, "w") as f:
          f.write("step,time,lr\n")

  def log_train_ce(self, niter, ce):
    """Writes training CE."""
    if self._write_to_csv:
      with open(self.train_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), ce))

  def log_train_acc(self, niter, acc):
    """Writes training accuracy."""
    if self._write_to_csv:
      with open(self.trainval_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), acc))

  def log_valid_acc(self, niter, acc):
    """Writes validation accuracy."""
    if self._write_to_csv:
      with open(self.val_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), acc))

  def log_learn_rate(self, niter, lr):
    """Writes validation accuracy."""
    if self._write_to_csv:
      with open(self.lr_file_name, "a") as f:
        f.write("{:d},{:s},{:e}\n".format(
            niter + 1, datetime.datetime.now().isoformat(), lr))
