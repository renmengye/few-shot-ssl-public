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
import tensorflow as tf

from fewshot.utils import logger
from fewshot.utils.debug import debug_identity

log = logger.get()


def weight_variable(shape,
                    init_method=None,
                    dtype=tf.float32,
                    init_param=None,
                    wd=None,
                    name=None,
                    trainable=True,
                    seed=0):
  """Declares a variable.

  Args:
    shape: Shape of the weights, list of int.
    init_method: Initialization method, "constant" or "truncated_normal".
    init_param: Initialization parameters, dictionary.
    wd: Weight decay, float.
    name: Name of the variable, str.
    trainable: Whether the variable can be trained, bool.

  Returns:
    var: Declared variable.
  """
  log.info("Weight shape {}".format([int(ss) for ss in shape]))
  if dtype != tf.float32:
    log.warning("Not using float32, currently using {}".format(dtype))
  if init_method is None:
    initializer = tf.zeros_initializer(dtype=dtype)
  elif init_method == "truncated_normal":
    if "mean" not in init_param:
      mean = 0.0
    else:
      mean = init_param["mean"]
    if "stddev" not in init_param:
      stddev = 0.1
    else:
      stddev = init_param["stddev"]
    initializer = tf.truncated_normal_initializer(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)
    log.info("Truncated normal initialization stddev={}".format(stddev))
  elif init_method == "uniform_scaling":
    if "factor" not in init_param:
      factor = 1.0
    else:
      factor = init_param["factor"]
    initializer = tf.uniform_unit_scaling_initializer(
        factor=factor, seed=seed, dtype=dtype)
    log.info("Uniform initialization factor={}".format(factor))
  elif init_method == "constant":
    if "val" not in init_param:
      value = 0.0
    else:
      value = init_param["val"]
    initializer = tf.constant_initializer(value)
    log.info("Constant initialization value={}".format(value))
  elif init_method == "numpy":
    assert "val" in init_param
    value = init_param["val"]
    initializer = value
    shape = None
    log.info("NumPy initialization value={}".format(value))
  elif init_method == "xavier":
    initializer = tf.contrib.layers.xavier_initializer(
        uniform=False, seed=seed, dtype=dtype)
    log.info("Xavier initialization")
  else:
    raise ValueError("Non supported initialization method!")
  if wd is not None:
    if wd > 0.0:
      reg = lambda x: tf.multiply(tf.nn.l2_loss(x), wd)
      log.info("Weight decay {}".format(wd))
    else:
      log.warning("No weight decay")
      reg = None
  else:
    log.warning("No weight decay")
    reg = None
  with tf.device("/cpu:0"):
    var = tf.get_variable(
        name,
        shape,
        initializer=initializer,
        regularizer=reg,
        dtype=dtype,
        trainable=trainable)
  log.info(var.name)
  return var


def cnn(x,
        filter_size,
        strides,
        pool_fn,
        pool_size,
        pool_strides,
        act_fn,
        dtype=tf.float32,
        add_bias=True,
        wd=None,
        init_std=None,
        init_method=None,
        batch_norm=True,
        scope="cnn",
        trainable=True,
        is_training=True,
        keep_ema=False,
        ext_wts=None):
  """Builds a convolutional neural networks.
  Each layer contains the following operations:
    1) Convolution, y = w * x.
    2) Additive bias (optional), y = w * x + b.
    3) Activation function (optional), y = g( w * x + b ).
    4) Pooling (optional).

  Args:
    x: Input variable.
    filter_size: Shape of the convolutional filters, list of 4-d int.
    strides: Convolution strides, list of 4-d int.
    pool_fn: Pooling functions, list of N callable objects.
    pool_size: Pooling field size, list of 4-d int.
    pool_strides: Pooling strides, list of 4-d int.
    act_fn: Activation functions, list of N callable objects.
    add_bias: Whether adding bias or not, bool.
    wd: Weight decay, float.
    scope: Scope of the model, str.
  """
  num_layer = len(filter_size)
  h = x
  wt_dict = {}
  with tf.variable_scope(scope):
    for ii in range(num_layer):
      with tf.variable_scope("layer_{}".format(ii)):
        if init_method is not None and init_method[ii]:
          _init_method = init_method[ii]
        else:
          _init_method = "truncated_normal"
        if ext_wts is not None:
          w = ext_wts["w_" + str(ii)]
          if type(w) == np.ndarray:
            w = tf.constant(w)
            log.info("Found all weights from numpy array")
          else:
            log.info("Found all weights from tensors")
        else:
          w = weight_variable(
              filter_size[ii],
              dtype=dtype,
              init_method=_init_method,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)
        wt_dict["w_" + str(ii)] = w
        if add_bias:
          if ext_wts is not None:
            b = ext_wts["b_" + str(ii)]
            if type(b) == np.ndarray:
              b = tf.constant(b)
              log.info("Found all biases from numpy array")
            else:
              log.info("Found all biases from tensors")
          else:
            b = weight_variable(
                [filter_size[ii][3]],
                dtype=dtype,
                init_method="constant",
                init_param={"val": 0},
                name="b",
                trainable=trainable)
          wt_dict["b_" + str(ii)] = b
        h = tf.nn.conv2d(h, w, strides=strides[ii], padding="SAME", name="conv")

        if add_bias:
          h = tf.add(h, b, name="conv_bias")

        if batch_norm:
          # Batch normalization.
          n_out = int(h.get_shape()[-1])
          if ext_wts is not None:
            beta = ext_wts["beta_" + str(ii)]
            gamma = ext_wts["gamma_" + str(ii)]
            if "emean_" + str(ii) in ext_wts:
              emean = ext_wts["emean_" + str(ii)]
              evar = ext_wts["evar_" + str(ii)]
              assign_ema = True
            else:
              assign_ema = False
            if type(ext_wts["beta_" + str(ii)]) == np.ndarray:
              beta = tf.constant(ext_wts["beta_" + str(ii)])
              gamma = tf.constant(ext_wts["gamma_" + str(ii)])

              if assign_ema:
                emean = tf.constant(ext_wts["emean_" + str(ii)])
                evar = tf.constant(ext_wts["evar_" + str(ii)])
              log.info("Found all BN weights from numpy array")
            else:
              log.info("Found all BN weights from tensors")
          else:
            beta = weight_variable(
                [n_out],
                dtype=dtype,
                init_method="constant",
                init_param={"val": 0.0},
                name="beta")
            gamma = weight_variable(
                [n_out],
                dtype=dtype,
                init_method="constant",
                init_param={"val": 1.0},
                name="gamma")
            emean = weight_variable(
                [n_out],
                dtype=dtype,
                init_method="constant",
                init_param={"val": 0.0},
                name="ema_mean",
                trainable=False)
            evar = weight_variable(
                [n_out],
                dtype=dtype,
                init_method="constant",
                init_param={"val": 1.0},
                name="ema_var",
                trainable=False)
            assign_ema = True

          wt_dict["beta_" + str(ii)] = beta
          wt_dict["gamma_" + str(ii)] = gamma

          if assign_ema:
            wt_dict["emean_" + str(ii)] = emean
            wt_dict["evar_" + str(ii)] = evar

          if is_training:
            decay = 0.9
            mean, var = tf.nn.moments(h, [0, 1, 2], name="moments")
            if assign_ema:
              ema_mean_op = tf.assign_sub(emean, (emean - mean) * (1 - decay))
              ema_var_op = tf.assign_sub(evar, (evar - var) * (1 - decay))
              with tf.control_dependencies([ema_mean_op, ema_var_op]):
                h = tf.nn.batch_normalization(h, mean, var, beta, gamma, 1e-5)
            else:
              h = tf.nn.batch_normalization(h, mean, var, beta, gamma, 1e-5)
          else:
            h = tf.nn.batch_normalization(h, emean, evar, beta, gamma, 1e-5)

        if act_fn[ii] is not None:
          h = act_fn[ii](h, name="act")
          # h = tf.tanh(h, name="act")
        if pool_fn[ii] is not None:
          _height = int(h.get_shape()[1])
          _width = int(h.get_shape()[2])

          # For reproducing Jake's experiment.
          log.info('CNN {} {}'.format(_height, _width))
          if _height % 2 == 1:
            h = h[:, :_height - 1, :_width - 1, :]
            # h = h[:, 1:_height, 1:_width, :]
            _height = int(h.get_shape()[1])
            _width = int(h.get_shape()[2])
            log.info("After resize {} {}".format(_height, _width))

          h = pool_fn[ii](
              h,
              pool_size[ii],
              strides=pool_strides[ii],
              padding="SAME",
              name="pool")
          _height = int(h.get_shape()[1])
          _width = int(h.get_shape()[2])
          log.info("After pool {} {}".format(_height, _width))
  return h, wt_dict


def mlp(x,
        dims,
        is_training=True,
        act_fn=None,
        dtype=tf.float32,
        add_bias=True,
        wd=None,
        init_std=None,
        init_method=None,
        scope="mlp",
        dropout=None,
        trainable=True):
  """Builds a multi-layer perceptron.
    Each layer contains the following operations:
        1) Linear transformation, y = w^T x.
        2) Additive bias (optional), y = w^T x + b.
        3) Activation function (optional), y = g( w^T x + b )
        4) Dropout (optional)

    Args:
        x: Input variable.
        dims: Layer dimensions, list of N+1 int.
        act_fn: Activation functions, list of N callable objects.
        add_bias: Whether adding bias or not, bool.
        wd: Weight decay, float.
        scope: Scope of the model, str.
        dropout: Whether to apply dropout, None or list of N bool.
    """
  num_layer = len(dims) - 1
  h = x
  with tf.variable_scope(scope):
    for ii in range(num_layer):
      with tf.variable_scope("layer_{}".format(ii)):
        dim_in = dims[ii]
        dim_out = dims[ii + 1]

        if init_method is not None and init_method[ii]:
          w = weight_variable(
              [dim_in, dim_out],
              init_method=init_method[ii],
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)
        else:
          w = weight_variable(
              [dim_in, dim_out],
              init_method="truncated_normal",
              dtype=dtype,
              init_param={"mean": 0.0,
                          "stddev": init_std[ii]},
              wd=wd,
              name="w",
              trainable=trainable)

        if add_bias:
          b = weight_variable(
              [dim_out],
              init_method="constant",
              dtype=dtype,
              init_param={"val": 0.0},
              name="b",
              trainable=trainable)

        h = tf.matmul(h, w, name="linear")
        if add_bias:
          h = tf.add(h, b, name="linear_bias")
        if act_fn and act_fn[ii] is not None:
          h = act_fn[ii](h)
        if dropout is not None and dropout[ii]:
          log.info("Apply dropout 0.5")
          if is_training:
            keep_prob = 0.5
          else:
            keep_prob = 1.0
          h = tf.nn.dropout(h, keep_prob=keep_prob)
  return h


def layer_norm(x,
               gamma=None,
               beta=None,
               axes=[1],
               eps=1e-3,
               scope="ln",
               name="ln_out"):
  """Applies layer normalization.
    Collect mean and variances on x except the first dimension. And apply
    normalization as below:
        x_ = gamma * (x - mean) / sqrt(var + eps)

    Args:
        x: Input tensor, [B, ...].
        axes: Axes to collect statistics.
        gamma: Scaling parameter.
        beta: Bias parameter.
        eps: Denominator bias.
        return_mean: Whether to also return the computed mean.

    Returns:
        normed: Layer-normalized variable.
        mean: Mean used for normalization (optional).
    """
  with tf.variable_scope(scope):
    x_shape = [x.get_shape()[-1]]
    mean, var = tf.nn.moments(x, axes, name='moments', keep_dims=True)
    normed = (x - mean) / tf.sqrt(eps + var)
    if gamma is not None:
      normed *= gamma
    if beta is not None:
      normed += beta
    normed = tf.identity(normed, name=name)
  return normed


def get_ln_act(act, affine=True, eps=1e-3, scope="ln_act"):
  """Gets a layer-normalized activation function.
  Args:
    act: Activation function, callable object.
    affine: Whether to add affine transformation, bool.
    eps: Denominator bias, float.
    scope: Scope of the operation, str.
  """

  def _act(x, reuse=None, name="ln_out"):
    """Layer-normalized activation function.
    Args:
      x: Input tensor.
      reuse: Whether to reuse the parameters, bool.
      name: Name for the output tensor.
    Returns:
      normed: Output tensor.
    """
    with tf.variable_scope(scope + "_params", reuse=reuse):
      if affine:
        x_shape = [x.get_shape()[-1]]
        beta = weight_variable(
            x_shape,
            init_method="constant",
            init_param={"val": 0.0},
            name="beta")
        gamma = weight_variable(
            x_shape,
            init_method="constant",
            init_param={"val": 1.0},
            name="gamma")
      else:
        beta = None
        gamma = None
    x_normed = layer_norm(
        x, axes=[1], gamma=gamma, beta=beta, eps=eps, scope=scope, name=name)
    return act(x_normed)

  return _act


def concat(x, axis):
  if tf.__version__.startswith("0"):
    return tf.concat(axis, x)
  else:
    return tf.concat(x, axis=axis)


def split(x, num, axis):
  if tf.__version__.startswith("0"):
    return tf.split(axis, num, x)
  else:
    return tf.split(x, num, axis)


def gumbel_softmax(logits, temperature, dtype=tf.float32, seed=0):
  """Gumbel Softmax Layer."""
  log_alpha = tf.nn.log_softmax(logits)
  eps = 1e-7
  gumbel = -tf.log(-tf.log(
      tf.random_uniform(
          tf.shape(logits), minval=0, maxval=1 - eps, dtype=dtype, seed=seed) +
      eps))
  prob = tf.nn.softmax((log_alpha + gumbel) / temperature)
  return prob


def gumbel_sigmoid(logits, temperature, dtype=tf.float32, seed=0):
  """Gumbel Sigmoid Layer."""
  logits = concat([logits, tf.zeros_like(logits)], 1)
  prob = gumbel_softmax(logits, temperature, dtype=dtype, seed=seed)
  return prob[:, 0]


def round_st(prob, dtype=tf.float32, name=None):
  g = tf.get_default_graph()
  with g.gradient_override_map({"Round": "Identity"}):
    y = tf.round(prob)
    y.set_shape(prob.get_shape())
    return y


if tf.__version__.startswith("0"):
  tf.NoGradient("Round")


def round_blk(prob):
  return tf.round(prob)


def _hardmax(prob):
  """Hard max numpy function."""
  # prob += np.random.uniform(0, 1e-5, prob.shape)  # Break symmetry.
  # return np.equal(prob, np.max(prob, axis=1, keepdims=True)).astype(np.float32)
  # y = np.zeros(prob.shape, dtype=prob.dtype)
  idx = np.argmax(prob, axis=1)
  # print(idx)
  y = np.eye(prob.shape[1], dtype=prob.dtype)[idx]
  # print(y)
  assert y.sum() == prob.shape[0]
  return y


def hardmax_st(prob, dtype=tf.float32, name=None):
  """Implements hard max with straight-through gradient estimator."""
  g = tf.get_default_graph()
  assert dtype == tf.float32, "Assert Float32"
  with g.gradient_override_map({"PyFuncStateless": "Identity"}):
    y = tf.py_func(_hardmax, [prob], [dtype], stateful=False, name=name)[0]
    y.set_shape(prob.get_shape())
    return y


def hardmax_blk(prob, dtype=tf.float32, name=None):
  """Implements hard max with straight-through gradient estimator."""
  idx = tf.argmax(prob, axis=1)
  y = tf.one_hot(idx, tf.shape(prob)[1])
  return y


def _categorical(prob):
  y = np.zeros(prob.shape, dtype=prob.dtype)
  # overflow = np.maximum((prob.sum(axis=1, keepdims=True) - 0.99), 0.0)
  prob = prob / prob.sum(axis=1, keepdims=True) * 0.99
  for ii in range(prob.shape[0]):
    y[ii] = np.random.multinomial(1, prob[ii])
  assert y.sum() == prob.shape[0]
  return y


def categorical_st(prob, dtype=tf.float32, name=None):
  g = tf.get_default_graph()
  assert dtype == tf.float32, "Assert Float32"
  with g.gradient_override_map({"PyFuncStateless": "Identity"}):
    y = tf.py_func(_categorical, [prob], [dtype], stateful=False, name=name)[0]
    y.set_shape(prob.get_shape())
    return y


def categorical_blk(prob, dtype=tf.float32, name=None, seed=0):
  idx = tf.multinomial(prob, 1, seed=seed)[:, 0]
  y = tf.one_hot(idx, int(prob.get_shape()[1]))
  return y


def sigmoid_cross_entropy_with_logits(logits, labels):
  if tf.__version__.startswith("0"):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, labels)
  else:
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
