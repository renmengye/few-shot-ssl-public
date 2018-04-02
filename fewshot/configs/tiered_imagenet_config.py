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
from fewshot.configs.config_factory import RegisterConfig


@RegisterConfig("tiered-imagenet", "basic")
class BasicConfig(object):
  """Standard CNN with prototypical layer."""

  def __init__(self):
    self.name = "tiered-imagenet_basic"
    self.model_class = "basic"
    self.height = 84
    self.width = 84
    self.num_channel = 3
    self.steps_per_valid = 2000
    self.steps_per_log = 100
    self.steps_per_save = 2000
    self.filter_size = [[3, 3, 3, 64]] + [[3, 3, 64, 64]] * 3
    self.strides = [[1, 1, 1, 1]] * 4
    self.pool_fn = ["max_pool"] * 4
    self.pool_size = [[1, 2, 2, 1]] * 4
    self.pool_strides = [[1, 2, 2, 1]] * 4
    self.conv_act_fn = ["relu"] * 4
    self.conv_init_method = None
    self.conv_init_std = [1.0e-2] * 4
    self.wd = 5e-5
    self.learn_rate = 1e-3
    self.normalization = "batch_norm"
    self.lr_scheduler = "fixed"
    self.max_train_steps = 200000
    self.lr_decay_steps = list(range(0, self.max_train_steps, 25000)[1:])
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x, range(
            len(self.lr_decay_steps))))
    self.similarity = "euclidean"


@RegisterConfig("tiered-imagenet", "basic-pretrain")
class BasicPretrainConfig(BasicConfig):

  def __init__(self):
    super(BasicPretrainConfig, self).__init__()
    self.max_train_steps = 4000
    self.lr_decay_steps = [2000, 2500, 3000, 3500]
    self.lr_list = list(
        map(lambda x: self.learn_rate * (0.5)**x,
            range(1,
                  len(self.lr_decay_steps) + 1)))
    self.similarity = "euclidean"


@RegisterConfig("tiered-imagenet", "kmeans-refine")
class KMeansRefineConfig(BasicConfig):

  def __init__(self):
    super(KMeansRefineConfig, self).__init__()
    self.name = "tiered-imagenet_kmeans-refine"
    self.model_class = "kmeans-refine"
    self.num_cluster_steps = 1


@RegisterConfig("tiered-imagenet", "kmeans-refine-radius")
class KMeansRefineDistractorConfig(BasicConfig):

  def __init__(self):
    super(KMeansRefineDistractorConfig, self).__init__()
    self.name = "tiered-imagenet_kmeans-refine-radius"
    self.model_class = "kmeans-refine-radius"
    self.num_cluster_steps = 1


@RegisterConfig("tiered-imagenet", "kmeans-refine-mask")
class KMeansRefineDistractorMSV3Config(BasicConfig):

  def __init__(self):
    super(KMeansRefineDistractorMSV3Config, self).__init__()
    self.name = "tiered-imagenet_kmeans-refine-mask"
    self.model_class = "kmeans-refine-mask"
    self.num_cluster_steps = 1
