import tensorflow as tf
import numpy as np
from . import hmc


#
# Theoretically optimal acceptance rate
TARGET_ACCEPTANCE_RATE = .65


def get_schedule(num, rad=4, for_calculating_marginal=True):
  """
  Calculate a temperature schedule for annealing.

  When calculating the marginal, we always include the prior at temperature (power) 1,
  only the temperature for the target varies.

  Returns:

    - a 2D numpy array. The first dimension indexes the schedule.
    Each temperature has 2 components, one for the prior, one for the target.
  """
  if num == 1:
    raise ValueError('Must have at least two temperatures')
  t = np.linspace(-rad, rad, num)
  s = 1.0 / (1.0 + np.exp(-t))
  beta = (s - np.min(s)) / (np.max(s) - np.min(s))
  schedule = np.array([1. - beta, beta]).T
  if for_calculating_marginal:
    # When calculating the marginal, we always include the prior at temperature (power) 1
    print('Calculating marginal')
    schedule[:, 0] = 1.
  return schedule


def temperature_pairs(schedule):
  """
  Calculate all consecutive pairs of temperatures.
  """
  return np.asarray([[schedule[i], schedule[i+1]] for i in range(len(schedule)-1)])


class HAIS(object):
  """
  Hamiltonian Annealed Importance Sampling
  """

  def __init__(self,
               prior,
               log_target,
               stepsize=.5,
               target_acceptance_rate=.65,
               avg_acceptance_slowness=0.9):
    """
    The model implements Hamiltonian Annealed Importance Sampling.
    Developed by @bilginhalil and @__Reidy__ on top of https://github.com/jiamings/ais/

    Example use case:
    logp(x|z) = |integrate over z|{logp(x|z,theta) + logp(z)}
    p(x|z, theta) -> likelihood function p(z) -> prior

    :param prior: The prior (or proposal distribution)
    :param log_target: Outputs log p(z) (up to a constant), it should take one parameter: z

    The following are parameters for HMC.
    :param stepsize:
    :param n_steps:
    :param target_acceptance_rate:
    :param avg_acceptance_slowness:
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
    self.target_acceptance_rate = target_acceptance_rate
    self.avg_acceptance_slowness = avg_acceptance_slowness

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
    smoothed_acceptance_rate = tf.constant(self.smoothed_acceptance_rate, shape=self.batch_shape, dtype=tf.float32)

    def condition(index, logw, z, v, smoothed_acceptance_rate):
      "The condition keeps the while loop going until we reach the end of the schedule."
      return tf.less(index, len(schedule) - 1)

    def body(index, logw, z, v, smoothed_acceptance_rate):
      "The body of the while loop over the schedule."
      #
      # Get the pair of temperatures for this transition
      t0 = tf.gather(schedule_tf, index)  # First temperature
      t1 = tf.gather(schedule_tf, index+1)  # Second temperature
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
          eps=self.stepsize
      )
      #
      # Smooth the acceptance rate
      smoothed_acceptance_rate = hmc.smooth_acceptance_rate(accept, smoothed_acceptance_rate, self.avg_acceptance_slowness)
      #
      return tf.add(index, 1), logw, znew, vnew, smoothed_acceptance_rate

    #
    # While loop across temperature schedule
    _, logw, z_i, v_i, smoothed_acceptance_rate = \
        tf.while_loop(condition, body, (i, logw, z0, v0, smoothed_acceptance_rate), parallel_iterations=1, swap_memory=True)
    with tf.control_dependencies([logw, smoothed_acceptance_rate]):
      return logw, z_i, smoothed_acceptance_rate


  def log_normalizer(self, logw, samples_axis):
    "The log of the mean (axis=0 for samples typically) of exp(log weights)"
    return tf.reduce_logsumexp(logw, axis=samples_axis) - tf.log(tf.cast(tf.shape(logw)[samples_axis], dtype=tf.float32))
