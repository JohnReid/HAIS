"""Implementation of Hamiltonian Annealed Importance Sampling (HAIS).

The implementation includes:

  - partial momentum refresh across HMC moves (the main idea of Sohl-Dickstein and Culpepper).
  - adaptive HMC step sizes to attempt to acheive an optimal acceptance rate.

"""

import tensorflow as tf
import numpy as np
from . import hmc


#
# Theoretically optimal acceptance rate
TARGET_ACCEPTANCE_RATE = .65


def get_schedule(T, r=4, for_calculating_marginal=True):
  """
  Calculate a temperature schedule for annealing.

  Evenly spaced points in :math:`[-r, r]` are pushed
  through the sigmoid function and affinely transformed to :math:`[0, 1]`.

  .. math::

    t_i &= (\\frac{2i}{T - 1} - 1) r, \\quad i = 0, \dots, T-1 \\\\
    s_i &= \\frac{1}{1+e^{-t_i}} \\\\
    \\beta_i &= \\frac{s_i - s_0}{s_{T-1} - s_0}

  Args:
    T: number of annealed densities (temperatures).
    r: defines the domain of the sigmoid.
    for_calculating_marginal: If True, the temperatures for the prior
      will all be 1. Otherwise, they will be :math:`1 - \\beta_i`.

  Returns:
    2D numpy array: A numpy array with shape `(2, T)`, the first dimension indexes the prior
    and the target, the second dimension indexes the annealed densities.
    Each temperature has 2 components, one for the prior, one for the target.
    The temperatures for the target are defined by the :math:`\\beta_i` above,
    i.e. `schedule[1, i]` :math:`= \\beta_i`.
    When calculating the marginal, we always include the prior at temperature 1,
    i.e. `schedule[0, i] = 1` for all :math:`i`,
    only the temperature for the target varies (from 0 to 1). This effectively
    makes the target distribution proportional to the posterior.

  """
  if T == 1:
    raise ValueError('Must have at least two temperatures')
  t = np.linspace(-r, r, T)
  s = 1.0 / (1.0 + np.exp(-t))
  beta = (s - np.min(s)) / (np.max(s) - np.min(s))
  schedule = np.array([1. - beta, beta]).T
  if for_calculating_marginal:
    # When calculating the marginal, we always include the prior at temperature (power) 1
    print('Calculating marginal')
    schedule[:, 0] = 1.
  return schedule


class HAIS(object):
  """
  An implementation of Hamiltonian Annealed Importance Sampling (HAIS).
  """

  def __init__(self,
               prior,
               log_target,
               stepsize=.5,
               smthd_acceptance_decay=0.9,
               adapt_stepsize = False,
               target_acceptance_rate=.65,
               stepsize_dec = .9,
               stepsize_inc = 1./.9,
               stepsize_min = 1e-5,
               stepsize_max = 1e3):
    """
    Example use case:
    :math:`\\log p(x|z) = \\int \\log p(x|z,\\theta) + \\log p(z) dz`
    :math:`p(x|z, theta) -> likelihood function p(z) -> prior`

    Args:
      prior: The prior (or proposal distribution).
      log_target: Function f(z) that returns a tensor to evaluate :math:`\\log p(z)` (up to a constant).
        When estimating the marginal likelihood, this function should calculate the likelihood.
      stepsize: HMC step size.
      smthd_acceptance_decay: The decay used when smoothing the acceptance rates.
      adapt_stepsize: If true the algorithm will adapt the step size for each chain to encourage
        the smoothed acceptance rate to approach a target acceptance rate.
      target_acceptance_rate: If adapting step sizes, the target smoothed acceptance rate. 0.65 is
        near the theoretical optimum, see "MCMC Using Hamiltonian Dynamics" by Radford Neal in the
        "Handbook of Monte Carlo" (2011).
      stepsize_dec: The scaling factor by which to reduce the step size if the acceptance rate is too low.
        Only used when adapting step sizes.
      stepsize_inc: The scaling factor by which to increase the step size if the acceptance rate is too high.
        Only used when adapting step sizes.
      stepsize_min: A hard lower bound on the step size.
        Only used when adapting step sizes.
      stepsize_max: A hard upper bound on the step size.
        Only used when adapting step sizes.
    """
    #
    # Model
    self.log_target = log_target
    self.prior = prior
    #
    # Dimensions
    self.batch_shape = self.prior.batch_shape
    self.event_shape = self.prior.event_shape
    self.shape = self.batch_shape.concatenate(self.event_shape)
    self.event_axes = list(range(len(self.batch_shape), len(self.shape)))
    #
    # HMC
    self.stepsize = stepsize
    self.smoothed_acceptance_rate = target_acceptance_rate
    self.smthd_acceptance_decay = smthd_acceptance_decay
    self.adapt_stepsize = adapt_stepsize
    self.target_acceptance_rate = target_acceptance_rate
    self.stepsize_dec = stepsize_dec
    self.stepsize_inc = stepsize_inc
    self.stepsize_min = stepsize_min
    self.stepsize_max = stepsize_max

  def log_f_i(self, z, t):
    "Unnormalized log density for intermediate distribution :math:`f_i`"
    return - self.energy_fn(z, t)

  def energy_fn(self, z, t):
    """
    Calculate the energy for each sample z at the temperature t. The temperature
    is a pair of temperatures, one for the prior and one for the target.
    """
    assert z.shape == self.shape
    prior_t = tf.gather(t, 0)
    prior_energy = prior_t * self.prior.log_prob(z)
    assert prior_energy.shape == self.batch_shape
    target_t = tf.gather(t, 1)
    likelihood_energy = target_t * self.log_target(z)
    return - prior_energy - likelihood_energy

  def ais(self, schedule):
    """
    Perform annealed importance sampling.

    Args:
        schedule: temperature schedule i.e. :math:`p(z)p(x|z)^t`
    """
    #
    # Convert the schedule into consecutive pairs of temperatures and their index
    schedule_tf = tf.convert_to_tensor(schedule, dtype=tf.float32)
    #
    # These are the variables that are passed to body() and condition() in the while loop
    i = tf.constant(0)
    logw = tf.zeros(self.batch_shape)
    z0 = self.prior.sample()
    v0 = tf.random_normal(tf.shape(z0))
    if self.adapt_stepsize:
      eps0 = tf.constant(self.stepsize, shape=self.batch_shape, dtype=tf.float32)
    else:
      eps0 = tf.constant(self.stepsize, dtype=tf.float32)
    smoothed_acceptance_rate = tf.constant(self.smoothed_acceptance_rate, shape=self.batch_shape, dtype=tf.float32)

    def condition(index, logw, z, v, eps, smoothed_acceptance_rate):
      "The condition keeps the while loop going until we reach the end of the schedule."
      return tf.less(index, len(schedule) - 1)

    def body(index, logw, z, v, eps, smoothed_acceptance_rate):
      "The body of the while loop over the schedule."
      #
      # Get the pair of temperatures for this transition
      t0 = tf.gather(schedule_tf, index)  # First temperature
      t1 = tf.gather(schedule_tf, index + 1)  # Second temperature
      #
      # Calculate u at the new temperature and at the old one
      new_u = self.log_f_i(z, t1)
      prev_u = self.log_f_i(z, t0)
      #
      # Add the difference in u to the weight
      logw = tf.add(logw, new_u - prev_u)
      #
      # New step: make a HMC move
      # print('z: {}'.format(z.shape))
      assert z.shape == self.shape
      accept, znew, vnew = hmc.hmc_move(
          z,
          v,
          lambda z: self.energy_fn(z, t1),
          event_axes=self.event_axes,
          eps=eps
      )
      #
      # Smooth the acceptance rate
      smoothed_acceptance_rate = hmc.smooth_acceptance_rate(
          accept, smoothed_acceptance_rate, self.smthd_acceptance_decay)
      #
      # Adaptive step size
      if self.adapt_stepsize:
        epsnew = self.adapt_step_size(eps, smoothed_acceptance_rate)
      else:
        epsnew = eps
      #
      return tf.add(index, 1), logw, znew, vnew, epsnew, smoothed_acceptance_rate

    #
    # While loop across temperature schedule
    _, logw, z_i, v_i, eps_i, smoothed_acceptance_rate = \
        tf.while_loop(
            condition, body, (i, logw, z0, v0, eps0, smoothed_acceptance_rate),
            parallel_iterations=1, swap_memory=True)
    #
    # Return weights, samples, step sizes and acceptance rates
    with tf.control_dependencies([logw, smoothed_acceptance_rate]):
      return logw, z_i, eps_i, smoothed_acceptance_rate

  def adapt_step_size(self, eps, smoothed_acceptance_rate):
    """Adapt the step size to adjust the smoothed acceptance rate to a theoretical optimum.
    """
    # print('stepsize_inc: {}'.format(stepsize_inc.shape))
    # print('stepsize_dec: {}'.format(stepsize_dec.shape))
    epsadapted = tf.where(
        smoothed_acceptance_rate > self.target_acceptance_rate,
        tf.constant(self.stepsize_inc, shape=smoothed_acceptance_rate.shape),
        tf.constant(self.stepsize_dec, shape=smoothed_acceptance_rate.shape)) * eps
    #
    # Make sure we stay within specified step size range
    epsadapted = tf.clip_by_value(epsadapted, clip_value_min=self.stepsize_min, clip_value_max=self.stepsize_max)
    #
    return epsadapted

  def log_normalizer(self, logw, samples_axis):
    "The log of the mean (axis=0 for samples typically) of exp(log weights)"
    return tf.reduce_logsumexp(logw, axis=samples_axis) \
        - tf.log(tf.cast(tf.shape(logw)[samples_axis], dtype=tf.float32))
