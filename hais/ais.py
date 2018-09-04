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


def get_schedule(T, r=4):
  """
  Calculate a temperature schedule for annealing.

  Evenly spaced points in :math:`[-r, r]` are pushed
  through the sigmoid function and affinely transformed to :math:`[0, 1]`.

  .. math::

    t_i &= (\\frac{2i}{T} - 1) r, \\quad i = 0, \dots, T \\\\
    s_i &= \\frac{1}{1+e^{-t_i}} \\\\
    \\beta_i &= \\frac{s_i - s_0}{s_T - s_0}

  Args:
    T: number of annealing transitions (number of temperatures + 1).
    r: defines the domain of the sigmoid.

  Returns:
    1-D numpy array: A numpy array with shape `(T+1,)` that
    monotonically increases from 0 to 1 (the values are the
    :math:`\\beta_i`).

  """
  if T == 1:
    raise ValueError('Must have at least two temperatures')
  t = np.linspace(-r, r, T)
  s = 1.0 / (1.0 + np.exp(-t))
  beta = (s - np.min(s)) / (np.max(s) - np.min(s))
  return beta


class HAIS(object):
  """
  An implementation of Hamiltonian Annealed Importance Sampling (HAIS).
  """

  def __init__(self,
               proposal=None,
               log_target=None,
               prior=None,
               log_likelihood=None,
               stepsize=.5,
               smthd_acceptance_decay=0.9,
               adapt_stepsize = False,
               target_acceptance_rate=.65,
               stepsize_dec = .9,
               stepsize_inc = 1./.9,
               stepsize_min = 1e-5,
               stepsize_max = 1e3):
    """
    Initialise the HAIS class.

    The proposal and target distribution must be specified in one of two ways:

      - *either* a `proposal` distribution :math:`q(x)` and unnormalised `log_target`
        density :math:`p(x)` should be supplied. In this case the `i`'th annealed density will be
        :math:`q(x)^{1-\\beta_i}p(x)^{\\beta_i}`
      - *or* a `prior` distribution :math:`q(x)` and normalised `log_likelihood` density :math:`p(x)` should
        be supplied. In this case the `i`'th annealed density will be
        :math:`q(x)p(x)^{\\beta_i}`


    Args:
      proposal: The proposal distribution.
      log_target: Function that returns a tensor evaluating :math:`\\log p(x)` (up to a constant).
      prior: The prior distribution.
      log_likelihood: Function that returns a tensor evaluating :the normalised log likelihood of :math:`x`.
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
    a = None
    b = 1
    (a is None) ^ (b is None)
    #
    # Check the arguments, either proposal and log_target should be supplied OR prior and log_likelihood
    # but not both
    if (proposal is None) == (prior is None):
      raise ValueError('Exactly one of the proposal and prior arguments should be supplied.')
    if (proposal is None) != (log_target is None):
      raise ValueError('Either both of the proposal and log_target arguments should be supplied or neither.')
    if (prior is None) != (log_likelihood is None):
      raise ValueError('Either both of the prior and log_likelihood arguments should be supplied or neither.')
    #
    # Model
    self.proposal = proposal
    self.log_target = log_target
    self.prior = prior
    self.log_likelihood = log_likelihood
    if self.proposal is None:
      self.q = self.prior
    else:
      self.q = self.proposal
    #
    # Dimensions
    self.batch_shape = self.q.batch_shape
    self.event_shape = self.q.event_shape
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

  def _log_f_i(self, z, beta):
    "Unnormalized log density for intermediate distribution :math:`f_i`"
    return - self._energy_fn(z, beta)

  def _energy_fn(self, z, beta):
    """
    Calculate the energy for each sample z at the temperature beta. The temperature
    is a pair of temperatures, one for the prior and one for the target.
    """
    assert z.shape == self.shape
    if self.proposal is None:
      prior_energy = self.prior.log_prob(z)
      target_energy = beta * self.log_likelihood(z)
    else:
      prior_energy = (1 - beta) * self.proposal.log_prob(z)
      target_energy = beta * self.log_target(z)
    assert prior_energy.shape == self.batch_shape
    assert target_energy.shape == self.batch_shape
    return - prior_energy - target_energy

  def ais(self, schedule):
    """
    Perform annealed importance sampling.

    Args:
        schedule: temperature schedule
    """
    #
    # Convert the schedule into consecutive pairs of temperatures and their index
    schedule_tf = tf.convert_to_tensor(schedule, dtype=tf.float32)
    #
    # These are the variables that are passed to body() and condition() in the while loop
    i = tf.constant(0)
    logw = tf.zeros(self.batch_shape)
    z0 = self.q.sample()
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
      new_u = self._log_f_i(z, t1)
      prev_u = self._log_f_i(z, t0)
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
          lambda z: self._energy_fn(z, t1),
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
        epsnew = self._adapt_step_size(eps, smoothed_acceptance_rate)
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

  def _adapt_step_size(self, eps, smoothed_acceptance_rate):
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
    """The log of the mean (over the `samples_axis`) of :math:`e^{logw}`
    """
    return tf.reduce_logsumexp(logw, axis=samples_axis) \
        - tf.log(tf.cast(tf.shape(logw)[samples_axis], dtype=tf.float32))
