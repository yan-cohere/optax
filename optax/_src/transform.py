# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient transformations."""

import functools
from typing import NamedTuple, Optional

import chex
import jax
from jax import tree_util as jtu
import jax.numpy as jnp

from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax.transforms import _accumulation
from optax.transforms import _adding


abs_sq = numerics.abs_sq


def _reject_complex(params):
  if any(jnp.iscomplexobj(x) for x in jtu.tree_leaves(params)):
    raise ValueError('This transformation does not support complex parameters.')


class ScaleByRssState(NamedTuple):
  """State holding the sum of gradient squares to date."""
  sum_of_squares: base.Updates


def scale_by_rss(
    initial_accumulator_value: float = 0.1,
    eps: float = 1e-7
) -> base.GradientTransformation:
  """Rescale updates by the root of the sum of all squared gradients to date.

  References:
    [Duchi et al, 2011](https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    [McMahan et al., 2010](https://arxiv.org/abs/1002.4908)

  Args:
    initial_accumulator_value: Starting value for accumulators, must be >= 0.
    eps: A small floating point value to avoid zero denominator.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    return ScaleByRssState(
        sum_of_squares=otu.tree_full_like(params, initial_accumulator_value))

  def update_fn(updates, state, params=None):
    del params
    sum_of_squares = jtu.tree_map(
        lambda g, t: abs_sq(g) + t, updates, state.sum_of_squares)
    inv_sqrt_g_square = jtu.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), sum_of_squares)
    updates = otu.tree_mul(inv_sqrt_g_square, updates)
    return updates, ScaleByRssState(sum_of_squares=sum_of_squares)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRmsState(NamedTuple):
  """State for exponential root mean-squared (RMS)-normalized updates."""
  nu: base.Updates


def scale_by_rms(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.
) -> base.GradientTransformation:
  r"""Rescale updates by the root of the exp. moving avg of the square.

  WARNING: PyTorch and optax's RMSprop implementations differ and could impact
    performance. In the denominator, optax uses $\sqrt{v + \epsilon}$ whereas
    PyTorch uses $\sqrt{v} + \epsilon$. See
    https://github.com/google-deepmind/optax/issues/532 for more detail.

  References:
    [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  Args:
    decay: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    initial_scale: Initial value for second moment.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    nu = otu.tree_full_like(params, initial_scale)  # second moment
    return ScaleByRmsState(nu=nu)

  def update_fn(updates, state, params=None):
    del params
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, decay, 2)
    updates = jtu.tree_map(
        lambda g, n: g * jax.lax.rsqrt(n + eps), updates, nu)
    return updates, ScaleByRmsState(nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRStdDevState(NamedTuple):
  """State for centered exponential moving average of squares of updates."""
  mu: base.Updates
  nu: base.Updates


def scale_by_stddev(
    decay: float = 0.9,
    eps: float = 1e-8,
    initial_scale: float = 0.
) -> base.GradientTransformation:
  """Rescale updates by the root of the centered exp. moving average of squares.

  References:
    [Hinton](www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

  Args:
    decay: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    initial_scale: Initial value for second moment.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    nu = otu.tree_full_like(params, initial_scale)  # second moment
    return ScaleByRStdDevState(mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, decay, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, decay, 2)
    updates = jtu.tree_map(
        lambda g, m, n: g * jax.lax.rsqrt(n - abs_sq(m) + eps),
        updates, mu, nu)
    return updates, ScaleByRStdDevState(mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByAdamState(NamedTuple):
  """State for the Adam algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_adam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False
) -> base.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  References:
    Kingma et al, `Adam: A Method for Stochastic Optimization
    <https://arxiv.org/abs/1412.6980>`_, 2014

    Dozat, `Incorporating Nesterov Momentum into Adam
    <https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ>`_ 2016

  .. warning::
    PyTorch and optax's adam follow Algorithm 1 of the Kingma
    and Ba's Adam paper, if reproducing old results note that TensorFlow
    used instead the formulation just before Section 2.1 of the paper.
    See https://github.com/deepmind/optax/issues/571 for more detail.

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.
    nesterov: Whether to use Nesterov momentum. The variant of Adam with
      Nesterov momentum is described in [Dozat 2016]

  Returns:
    A `GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    if nesterov:
      mu_hat = jtu.tree_map(
          lambda m, g: b1 * m + (1 - b1) * g,
          otu.tree_bias_correction(
              mu, b1, numerics.safe_int32_increment(count_inc)),
          otu.tree_bias_correction(updates, b1, count_inc))
    else:
      mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    # Dozat 2016 https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ
    # Algorithm 2 further multiplies Adam's standard nu_hat by b2. It is
    # unclear why. Other Nadam implementations also omit the extra b2 factor.
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jtu.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByAmsgradState(NamedTuple):
  """State for the AMSGrad algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates
  nu_max: base.Updates


def scale_by_amsgrad(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the AMSGrad algorithm.

  References:
    [Reddi et al, 2018](https://openreview.net/forum?id=ryQu7f-RZ)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    mu_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    nu_max = otu.tree_zeros_like(params)
    return ScaleByAmsgradState(
        count=jnp.zeros([], jnp.int32),
        mu=mu, nu=nu, nu_max=nu_max)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    nu_max = jtu.tree_map(jnp.maximum, state.nu_max, nu_hat)
    updates = jtu.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_max)
    mu = otu.tree_cast(mu, mu_dtype)
    return updates, ScaleByAmsgradState(
        count=count_inc,
        mu=mu, nu=nu, nu_max=nu_max)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_adamax(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8
) -> base.GradientTransformation:
  """Rescale updates according to the Adamax algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted maximum of grads.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    nu = otu.tree_zeros_like(params)  # Infinite moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    count_inc = numerics.safe_int32_increment(state.count)
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_infinity_moment(updates, state.nu, b2, eps)
    # Bias correction for mean. No bias correction needed for infinity moment.
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    updates = jtu.tree_map(lambda m, v: m / v, mu_hat, nu)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByLionState(NamedTuple):
  """State for the Lion algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates


def scale_by_lion(
    b1: float = 0.9,
    b2: float = 0.99,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the Lion algorithm.

  References:
    [Chen et al, 2023](https://arxiv.org/abs/2302.06675)

  Args:
    b1: Rate for combining the momentum and the current grad.
    b2: Decay rate for the exponentially weighted average of grads.
    mu_dtype: Optional `dtype` to be used for the momentum; if
      `None` then the `dtype is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # moment
    return ScaleByLionState(count=jnp.zeros([], jnp.int32), mu=mu)

  def update_fn(updates, state, params=None):
    del params
    updates_new = jtu.tree_map(
        lambda g, m: jnp.sign((1. - b1) * g + b1 * m), updates, state.mu)
    mu = otu.tree_update_moment(updates, state.mu, b2, 1)
    mu = otu.tree_cast(mu, mu_dtype)
    count_inc = numerics.safe_int32_increment(state.count)
    return updates_new, ScaleByLionState(count=count_inc, mu=mu)

  return base.GradientTransformation(init_fn, update_fn)


def scale(
    step_size: float
) -> base.GradientTransformation:
  """Scale updates by some fixed scalar `step_size`.

  Args:
    step_size: A scalar corresponding to a fixed scaling factor for updates.

  Returns:
    A `GradientTransformation` object.
  """

  def update_fn(updates, state, params=None):
    del params
    updates = jtu.tree_map(lambda g: step_size * g, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def scale_by_param_block_norm(
    min_scale: float = 1e-3
) -> base.GradientTransformation:
  """Scale updates for each param block by the norm of that block's parameters.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    min_scale: Minimum scaling factor.

  Returns:
    A `GradientTransformation` object.
  """

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jtu.tree_map(
        lambda u, p: u * numerics.safe_norm(p, min_scale),
        updates, params)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def scale_by_param_block_rms(
    min_scale: float = 1e-3
) -> base.GradientTransformation:
  """Scale updates by rms of the gradient for each param vector or matrix.

  A `block` is here a weight vector (e.g. in a Linear layer) or a weight matrix
  (e.g. in a convolutional layer) appearing as a leaf in the grads/param pytree.

  Args:
    min_scale: Minimum scaling factor.

  Returns:
    A `GradientTransformation` object.
  """

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)
    updates = jtu.tree_map(
        lambda u, p: u * numerics.safe_root_mean_squares(p, min_scale),
        updates, params)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


class ScaleByAdaDeltaState(NamedTuple):
  """State for the rescaling by Adadelta algoritm."""

  e_g: base.Updates
  e_x: base.Updates


def scale_by_adadelta(
    rho: float = 0.9, eps: float = 1e-6
) -> base.GradientTransformation:
  """Rescale updates according to the Adadelta algorithm.

  References:
    [Matthew D. Zeiler, 2012](https://arxiv.org/pdf/1212.5701.pdf)

  Args:
    rho: A coefficient used for computing a running average of squared
      gradients.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    e_g = otu.tree_zeros_like(params)  # E[squared gradient]
    e_x = otu.tree_zeros_like(params)  # E[squared update]
    return ScaleByAdaDeltaState(e_g=e_g, e_x=e_x)

  def update_fn(updates, state, params=None):
    del params
    e_g = otu.tree_update_moment(updates, state.e_g, rho, 2)
    updates = jtu.tree_map(
        lambda g, cur_e_g, prev_e_x: (
            jnp.sqrt(prev_e_x + eps) / jnp.sqrt(cur_e_g + eps)
        )
        * g,
        updates,
        e_g,
        state.e_x,
    )
    e_x = otu.tree_update_moment(updates, state.e_x, rho, 2)
    return updates, ScaleByAdaDeltaState(e_g=e_g, e_x=e_x)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByBeliefState(NamedTuple):
  """State for the rescaling by AdaBelief algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates


def scale_by_belief(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16
) -> base.GradientTransformation:
  """Rescale updates according to the AdaBelief algorithm.

  References:
    [Zhuang et al, 2020](https://arxiv.org/abs/2010.07468)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of variance of grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the second moment of the prediction error to
      improve numerical stability. If backpropagating gradients through the
      gradient transformation (e.g. for meta-learning), this must be non-zero.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    s = otu.tree_zeros_like(params)  # Second Central moment
    return ScaleByBeliefState(count=jnp.zeros([], jnp.int32), mu=mu, nu=s)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    prediction_error = jtu.tree_map(
        lambda g, m: g-m, updates, state.mu)
    nu = otu.tree_update_moment_per_elem_norm(prediction_error, state.nu, b2, 2)
    nu = jtu.tree_map(lambda v: v + eps_root, nu)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jtu.tree_map(
        lambda m, v: m / (jnp.sqrt(v) + eps), mu_hat, nu_hat)
    return updates, ScaleByBeliefState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_yogi(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-3,
    eps_root: float = 0.0,
    initial_accumulator_value: float = 1e-6
) -> base.GradientTransformation:
  """Rescale updates according to the Yogi algorithm.

  Supports complex numbers, see
  https://gist.github.com/wdphy16/118aef6fb5f82c49790d7678cf87da29

  References:
    [Zaheer et al, 2018](https://papers.nips.cc/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html) #pylint:disable=line-too-long

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of variance of grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    initial_accumulator_value: The starting value for accumulators.
      Only positive values are allowed.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    mu = otu.tree_full_like(params, initial_accumulator_value)  # First moment
    nu = otu.tree_full_like(params, initial_accumulator_value)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = jtu.tree_map(
        lambda g, v: v - (1 - b2) * jnp.sign(v - abs_sq(g)) * abs_sq(g),
        updates, state.nu)
    count_inc = numerics.safe_int32_increment(state.count)
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jtu.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_radam(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    threshold: float = 5.0
) -> base.GradientTransformation:
  """Rescale updates according to the Rectified Adam algorithm.

  References:
    [Liu et al, 2020](https://arxiv.org/abs/1908.03265)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    threshold: Threshold for variance tractability.

  Returns:
    A `GradientTransformation` object.
  """

  ro_inf = 2./(1 - b2) - 1
  def _radam_update(params):
    ro = params[0]
    mu_hat = params[1]
    nu_hat = params[2]
    r = jnp.sqrt((ro - 4)*(ro - 2)*ro_inf/((ro_inf - 4)*(ro_inf - 2)*ro))
    updates = jtu.tree_map(
        lambda m, v: r*m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
    return updates

  def init_fn(params):
    mu = otu.tree_zeros_like(params)  # First moment
    nu = otu.tree_zeros_like(params)  # Second moment
    return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def update_fn(updates, state, params=None):
    del params
    mu = otu.tree_update_moment(updates, state.mu, b1, 1)
    nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    b2t = b2**count_inc
    ro = ro_inf - 2 * count_inc * b2t / (1 - b2t)
    mu_hat = otu.tree_bias_correction(mu, b1, count_inc)
    nu_hat = otu.tree_bias_correction(nu, b2, count_inc)
    updates = jax.lax.cond(
        ro >= threshold, _radam_update, lambda _: mu_hat,
        (ro, mu_hat, nu_hat))
    return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByRpropState(NamedTuple):
  step_sizes: base.Updates
  prev_updates: base.Updates


def scale_by_rprop(
    learning_rate: float,
    eta_minus: float = 0.5,
    eta_plus: float = 1.2,
    min_step_size: float = 1e-6,
    max_step_size: float = 50.0,
) -> base.GradientTransformation:
  """Scale with the Rprop optimizer.

  Rprop, short for resillient backpropogation, is a first order variant of
  gradient descent. It responds only to the sign of the gradient by increasing
  or decreasing the step size selected per parameter exponentially to speed up
  convergence and avoid oscillations.

  References:
    PyTorch implementation:
      https://pytorch.org/docs/stable/generated/torch.optim.Rprop.html
    Riedmiller and Braun, 1993: https://ieeexplore.ieee.org/document/298623
    Igel and Hüsken, 2003:
      https://www.sciencedirect.com/science/article/abs/pii/S0925231201007007

  Args:
    learning_rate: The initial step size.
    eta_minus: Multiplicative factor for decreasing step size. This is applied
      when the gradient changes sign from one step to the next.
    eta_plus: Multiplicative factor for increasing step size. This is applied
      when the gradient has the same sign from one step to the next.
    min_step_size: Minimum allowed step size. Smaller steps will be clipped to
      this value.
    max_step_size: Maximum allowed step size. Larger steps will be clipped to
      this value.

  Returns:
    The corresponding `GradientTransformation`.
  """

  def init_fn(params):
    step_sizes = otu.tree_full_like(params, learning_rate)
    prev_updates = otu.tree_zeros_like(params)
    return ScaleByRpropState(step_sizes, prev_updates)

  def update_fn(updates, state, params=None):
    del params
    sign = jtu.tree_map(
        lambda g, prev_g: g * prev_g, updates, state.prev_updates)
    step_sizes = jtu.tree_map(
        lambda s, step_size: jnp.where(
            s == 0,
            step_size,
            jnp.clip(
                step_size * jnp.where(s > 0, eta_plus, eta_minus),
                a_min=min_step_size, a_max=max_step_size
            )
        ),
        sign, state.step_sizes
    )
    prev_updates = jtu.tree_map(
        lambda s, g, step_size: jnp.where(
            s < 0, jnp.zeros_like(g), step_size * jnp.sign(g)),
        sign, updates, step_sizes)
    updates = jtu.tree_map(
        lambda s, g, prev_g: jnp.where(s < 0, jnp.zeros_like(prev_g), prev_g),
        sign, prev_updates, state.prev_updates)
    return updates, ScaleByRpropState(step_sizes, prev_updates)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByScheduleState(NamedTuple):
  """Maintains count for scale scheduling."""
  count: chex.Array  # shape=(), dtype=jnp.int32


def scale_by_learning_rate(
    learning_rate: base.ScalarOrSchedule,
    *,
    flip_sign: bool = True,
) -> base.GradientTransformation:
  """Scale by the (negative) learning rate (either as scalar or as schedule).

  Args:
    learning_rate: Can either be a scalar or a schedule (i.e. a callable that
      maps an (int) step to a float).
    flip_sign: When set to True (the default) this corresponds to scaling by the
      negative learning rate.

  Returns:
    An optax.GradientTransformation that corresponds to multiplying the gradient
    with `-learning_rate` (if flip_sign is True) or with `learning_rate` (if
    flip_sign is False).
  """
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return scale_by_schedule(lambda count: m * learning_rate(count))
  return scale(m * learning_rate)


def scale_by_schedule(
    step_size_fn: base.Schedule
) -> base.GradientTransformation:
  """Scale updates using a custom schedule for the `step_size`.

  Args:
    step_size_fn: A function that takes an update count as input and proposes
      the step_size to multiply the updates by.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return ScaleByScheduleState(count=jnp.zeros([], jnp.int32))

  def update_fn(updates, state, params=None):
    del params
    step_size = step_size_fn(state.count)
    updates = jtu.tree_map(
        lambda g: jnp.array(step_size, dtype=g.dtype) * g, updates)
    return updates, ScaleByScheduleState(
        count=numerics.safe_int32_increment(state.count))

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_trust_ratio(
    min_norm: float = 0.0,
    trust_coefficient: float = 1.,
    eps: float = 0.,
) -> base.GradientTransformation:
  """Scale updates by `trust ratio`.

  References:
    [You et. al 2020](https://arxiv.org/abs/1904.00962)

  Args:
    min_norm: Minimum norm for params and gradient norms; by default is zero.
    trust_coefficient: A multiplier for the trust ratio.
    eps: Additive constant added to the denominator for numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError(base.NO_PARAMS_MSG)

    def _scale_update(update, param):

      # Clip norms to minimum value, by default no clipping.
      param_norm = numerics.safe_norm(param, min_norm)
      update_norm = numerics.safe_norm(update, min_norm)
      trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

      # If no minimum norm clipping is used
      # Set trust_ratio to 1 in case where parameters would never be updated.
      zero_norm = jnp.logical_or(param_norm == 0., update_norm == 0.)
      safe_trust_ratio = jnp.where(
          zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio)

      return update * safe_trust_ratio

    updates = jtu.tree_map(_scale_update, updates, params)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


class ApplyEvery(NamedTuple):
  """Contains a counter and a gradient accumulator."""
  count: chex.Array
  grad_acc: base.Updates


def apply_every(
    k: int = 1
) -> base.GradientTransformation:
  """Accumulate gradients and apply them every k steps.

  Note that if this transformation is part of a chain, the states of the other
  transformations will still be updated at every step. In particular, using
  `apply_every` with a batch size of N/2 and k=2 is not necessarily equivalent
  to not using `apply_every` with a batch size of N. If this equivalence is
  important for you, consider using the `optax.MultiSteps`.

  Args:
    k: Emit non-zero gradients every k steps, otherwise accumulate them.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    grad_acc = otu.tree_zeros_like(params)
    return ApplyEvery(count=jnp.zeros([], jnp.int32), grad_acc=grad_acc)

  def update_fn(updates, state, params=None):
    del params
    c = state.count % k
    acc = c != 0
    grad_acc = jtu.tree_map(
        lambda g, ga: acc * ga + g, updates, state.grad_acc)
    emit = c == (k - 1)
    updates = jtu.tree_map(lambda ga: emit * ga, grad_acc)
    count_inc = numerics.safe_int32_increment(state.count)
    return updates, ApplyEvery(count=count_inc % k, grad_acc=grad_acc)

  return base.GradientTransformation(init_fn, update_fn)


def _subtract_mean(g):
  if len(g.shape) > 1:
    return g - g.mean(tuple(range(1, len(g.shape))), keepdims=True)
  else:
    return g


CentralState = base.EmptyState


def centralize() -> base.GradientTransformation:
  """Centralize gradients.

  References:
    [Yong et al, 2020](https://arxiv.org/abs/2004.01461)

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    del params
    return CentralState()

  def update_fn(updates, state, params=None):
    del params
    updates = jtu.tree_map(_subtract_mean, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


class ScaleBySM3State(NamedTuple):
  """State for the SM3 algorithm."""
  mu: base.Updates
  nu: base.Updates


def scale_by_sm3(
    b1: float = 0.9,
    b2: float = 1.0,
    eps: float = 1e-8
) -> base.GradientTransformation:
  """Scale updates by `sm3`.

  References:
    [Anil et. al 2019](https://arxiv.org/abs/1901.11150)

  Args:
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps: Term added to the denominator to improve numerical stability.

  Returns:
    A `GradientTransformation` object.
  """

  def zeros_for_dim(p):
    return [jnp.zeros([s]) for s in p.shape]

  def init_fn(params):
    _reject_complex(params)
    mu = jtu.tree_map(zeros_for_dim, params)
    nu = otu.tree_zeros_like(params)
    return ScaleBySM3State(mu, nu)

  def _expanded_shape(shape, axis):
    # Replaces a `shape` of [M, N, K] with 1 in all dimensions except for i.
    # For eg: i = 1 returns [1, N, 1].
    rank = len(shape)
    return [1] * axis + [shape[axis]] + [1] * (rank - axis - 1)

  def _new_accum(g, v):
    coeffs = ((1.0 - b2) if b2 != 1.0 else 1.0, b2)
    if g.ndim < 2:
      return coeffs[0]*g**2 + coeffs[1]*v[0]
    else:
      return coeffs[0]*g**2 + coeffs[1]*functools.reduce(jnp.minimum, v)

  def _new_mu(g, i):
    if g.ndim < 2:
      return g
    else:
      return jnp.max(g, axis=other_axes(i, g.ndim))

  def other_axes(idx, ndim):
    return list(range(idx)) + list(range(idx+1, ndim))

  def update_fn(updates, state, params=None):
    del params
    mu = jtu.tree_map(
        lambda g, v:  # pylint:disable=g-long-lambda
        [jnp.reshape(v[i], _expanded_shape(g.shape, i)) for i in range(g.ndim)],
        updates, state.mu)
    accum = jtu.tree_map(_new_accum, updates, mu)
    accum_inv_sqrt = jtu.tree_map(
        lambda t: jnp.where(t > 0, jax.lax.rsqrt(t + eps), 0.0), accum)
    up = jtu.tree_map(lambda g, a: g*a, updates, accum_inv_sqrt)
    nu = otu.tree_update_moment(up, state.nu, b1, 1)
    mu = jtu.tree_map(
        lambda g: [_new_mu(g, i) for i in range(g.ndim)], accum)

    return nu, ScaleBySM3State(mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByNovogradState(NamedTuple):
  """State for Novograd."""
  count: chex.Array
  mu: base.Updates
  nu: base.Updates


def scale_by_novograd(
    b1: float = 0.9,
    b2: float = 0.25,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  """Computes NovoGrad updates.

  References:
    [Ginsburg et al, 2019](https://arxiv.org/abs/1905.11286)

  Args:
    b1: A decay rate for the exponentially weighted average of grads.
    b2: A decay rate for the exponentially weighted average of squared grads.
    eps: A term added to the denominator to improve numerical stability.
    eps_root: A term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    weight_decay: A scalar weight decay rate.
    mu_dtype: An optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """

  mu_dtype = utils.canonicalize_dtype(mu_dtype)

  def init_fn(params):
    mu = otu.tree_zeros_like(params, dtype=mu_dtype)  # First moment
    nu = jtu.tree_map(lambda _: 0.0, params)  # Second moment
    return ScaleByNovogradState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

  def nu_addition(grads):
    return jnp.linalg.norm(grads)**2

  def mu_addition(grads, params, nu):
    return grads / (jnp.sqrt(nu + eps_root) + eps) + weight_decay * params

  def init_nu(grads, nu):
    del nu
    return jtu.tree_map(nu_addition, grads)

  def update_nu(grads, nu):
    updates = jtu.tree_map(nu_addition, grads)
    return otu.tree_update_moment(updates, nu, b2, 1)

  def init_mu(grads, params, mu, nu):
    del mu
    return jtu.tree_map(mu_addition, grads, params, nu)

  def update_mu(grads, params, mu, nu):
    updates = jtu.tree_map(mu_addition, grads, params, nu)
    return jtu.tree_map(lambda m, u: b1 * m + u, mu, updates)

  def update_fn(updates, state, params):
    count_inc = numerics.safe_int32_increment(state.count)

    nu = jax.lax.cond(
        count_inc == 1, init_nu, update_nu, updates, state.nu)
    mu = jax.lax.cond(
        count_inc == 1, init_mu, update_mu, updates, params, state.mu, nu)

    mu = otu.tree_cast(mu, mu_dtype)
    updates = mu
    return updates, ScaleByNovogradState(count=count_inc, mu=mu, nu=nu)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_optimistic_gradient(
    alpha: float = 1.0,
    beta: float = 1.0
) -> base.GradientTransformation:
  """Compute generalized optimistic gradients.

  References:
    [Mokhtari et al, 2019](https://arxiv.org/abs/1901.08511v2)

  Args:
    alpha: Coefficient for generalized optimistic gradient descent.
    beta: Coefficient for negative momentum.

  Returns:
    A `GradientTransformation` object.
  """

  def init_fn(params):
    return TraceState(trace=otu.tree_zeros_like(params))

  def update_fn(updates, state, params=None):
    del params

    new_updates = jtu.tree_map(
        lambda grad_t, grad_tm1: (alpha + beta) * grad_t - beta * grad_tm1,
        updates, state.trace)
    return new_updates, TraceState(trace=updates)

  return base.GradientTransformation(init_fn, update_fn)


class ScaleByDistanceOverGradientsState(NamedTuple):
  """State for scale_by_distance_over_gradients."""

  max_dist: base.OptState
  grad_sum_of_squares: base.OptState
  init_params: base.OptState


def scale_by_distance_over_gradients(
    reps_rel=1e-6, eps=1e-8, param_dtype=jnp.float32, global_scale=1.0
) -> base.GradientTransformation:
  """Distance-over-gradients learning rate-free optimizer.

  This implementation stores a single copy of the model parameters, plus two
  scalars per parameter array. It is equivalent to "Layer-wise DoG" (LDoG)
  in the paper.

  The authors recommend using model averaging with this optimizer.

  References:
    ["DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size
    Schedule"](https://arxiv.org/pdf/2302.12022.pdf)

  Args:
    reps_rel: Used to compute initial learning rate. Recommended values are 1e-4
      for models using batch norm, 1e-6 otherwise.
    eps: Small loading term to avoid divide-by-zero errors.
    param_dtype: dtype for storing initial parameters.
    global_scale: Global scale factor, typically 1.0 or -1.0

  Returns:
    A `GradientTransformation` object.
  """

  def _l2(x, y=0.0):
    return jnp.sqrt(jnp.square(x - y).sum())

  def init_fn(params):
    return ScaleByDistanceOverGradientsState(
        # Initial distance (needed to prevent zero step sizes).
        jtu.tree_map(lambda x: reps_rel * (1 + _l2(x)), params),
        # Initial gradient sum-of-squares.
        jtu.tree_map(lambda x: jnp.zeros(1), params),
        # Initial params, cast to preferred precision.
        otu.tree_cast(params, param_dtype),
    )

  def update_fn(updates, state: ScaleByDistanceOverGradientsState, params):
    # update max distance
    max_dist = jtu.tree_map(
        lambda d, x, y: jnp.maximum(d, _l2(x, y)),
        state.max_dist,
        params,
        state.init_params,
    )

    # update gradient sum-of-squares
    g_sos = jtu.tree_map(
        lambda x, y: x + jnp.square(y).sum(), state.grad_sum_of_squares, updates
    )

    def _tx(g, d, g_sos):
      """Apply the transformation."""
      eta = global_scale * (d / jnp.sqrt(g_sos + eps))
      return eta * g

    updates = jtu.tree_map(_tx, max_dist, g_sos, updates)

    # new state
    state = ScaleByDistanceOverGradientsState(
        max_dist, g_sos, state.init_params
    )

    return updates, state

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_polyak(
    f_min: float = 0.0,
    max_learning_rate: float = 1.0,
    eps: float = 0.0,
) -> base.GradientTransformationExtraArgs:
  """Scales the update by Polyak's step-size."""

  def update_fn(
      updates: base.Updates,
      state: base.EmptyState,
      params: Optional[base.Params] = None,
      *,
      value: float,
      **extra_args,
  ) -> tuple[base.Updates, base.EmptyState]:
    """Scales the update by the Polyak step-size.

    Args:
      updates: the updates to be scaled.
      state: the state of the transformation.
      params: the parameters of the model.
      value: the value of the loss function.
      **extra_args: additional keyword arguments. They are ignored by this
        transformation.
    Returns:
      The scaled updates and the state of the transformation.
    """
    del params, extra_args
    grad_sq_norm = otu.tree_l2_norm(updates, squared=True)
    # avoid division by zero
    step = jnp.where(
        grad_sq_norm + eps <= jnp.finfo(float).eps,
        jnp.array(0.0),
        jnp.minimum(
            (value - f_min) / (grad_sq_norm + eps), max_learning_rate
        ),
    )
    updates = otu.tree_scalar_mul(step, updates)
    return updates, state

  return base.GradientTransformationExtraArgs(base.init_empty_state, update_fn)


### Legacy symbols to be removed. ###


@functools.partial(
    chex.warn_deprecated_function,
    replacement='optax.tree_utils.tree_cast')
def cast_tree(
    tree: chex.ArrayTree,
    dtype: Optional[chex.ArrayDType]
) -> chex.ArrayTree:
  return otu.tree_cast(tree, dtype)

trace = _accumulation.trace
TraceState = _accumulation.TraceState
ema = _accumulation.ema
EmaState = _accumulation.EmaState
add_noise = _adding.add_noise
AddNoiseState = _adding.AddNoiseState
add_decayed_weights = _adding.add_decayed_weights
AddDecayedWeightsState = base.EmptyState
ScaleState = base.EmptyState
ScaleByTrustRatioState = base.EmptyState
