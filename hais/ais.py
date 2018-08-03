import tensorflow as tf
import numpy as np
from .hmc import hmc_move, hmc_updates


def get_schedule(num, rad=4):
  """
  Calculate a temperature schedule for annealing.
  """
  if num == 1:
      return np.array([0.0, 1.0])  # Must have at least two temperatures
  t = np.linspace(-rad, rad, num)
  s = 1.0 / (1.0 + np.exp(-t))
  return (s - np.min(s)) / (np.max(s) - np.min(s))


class AIS(object):
  """
  Hamiltonian Annealed Importance Sampling
  """

  def __init__(self,
               x_ph,
               qz,
               log_likelihood_fn,
               z_dim,
               batch_size,
               num_samples,
               stepsize=0.01, n_steps=2,
               target_acceptance_rate=.65,
               avg_acceptance_slowness=0.9,
               stepsize_min=1e-8,
               stepsize_max=0.5,
               stepsize_dec=0.99,
               stepsize_inc=1.01):
    """
    The model implements Hamiltonian AIS.
    Developed by @bilginhalil and @__Reidy__ on top of https://github.com/jiamings/ais/

    Example use case:
    logp(x|z) = |integrate over z|{logp(x|z,theta) + logp(z)}
    p(x|z, theta) -> likelihood function p(z) -> prior

    :param x_ph: Placeholder for x
    :param qz: Proposal distribution, or equivalently the prior
    :param log_likelihood_fn: Outputs the logp(x|z, theta), it should take two parameters: x and z
    :param num_samples: Number of samples to sample from in order to estimate the likelihood.

    The following are parameters for HMC.
    :param stepsize:
    :param n_steps:
    :param target_acceptance_rate:
    :param avg_acceptance_slowness:
    :param stepsize_min:
    :param stepsize_max:
    :param stepsize_dec:
    :param stepsize_inc:
    """
    #
    # Dimensions
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.z_dim = z_dim
    #
    # Model
    self.log_likelihood_fn = log_likelihood_fn
    self.prior = qz
    #
    # Variables
    self.z_shape = [self.batch_size, self.num_samples, self.z_dim]
    self.x = tf.tile(tf.expand_dims(x_ph, axis=1), [1, self.num_samples, 1])
    #
    # HMC
    self.stepsize = stepsize
    self.avg_acceptance_rate = target_acceptance_rate
    self.n_steps = n_steps
    self.stepsize_min = stepsize_min
    self.stepsize_max = stepsize_max
    self.stepsize_dec = tf.constant(stepsize_dec, shape=[self.batch_size, self.num_samples])
    self.stepsize_inc = tf.constant(stepsize_inc, shape=[self.batch_size, self.num_samples])
    self.target_acceptance_rate = target_acceptance_rate
    self.avg_acceptance_slowness = avg_acceptance_slowness

  def log_f_i(self, z, t):
    "Unnormalized log density for intermediate distribution `f_i`"
    energy = self.energy_fn(z, t)
    return - energy

  def energy_fn(self, z, t):
    "Calculate the energy for each sample at the temperature."
    prior_energy = self.prior.log_prob(z)
    ll = self.log_likelihood_fn(self.x, z)
    likelihood_energy = t * ll
    # likelihood_energy: BATCH_SIZE x N_CHAINS
    return - prior_energy - likelihood_energy

  def ais(self, schedule):
    """
    Perform annealed importance sampling.

        :param schedule: temperature schedule i.e. `p(z)p(x|z)^t`
    """
    #
    # Convert the schedule into consecutive pairs of temperatures and their index
    temp_pairs = tf.unstack(
        tf.convert_to_tensor([[i, t0, t1] for i, (t0, t1) in enumerate(zip(schedule[:-1], schedule[1:]))]))
    #
    # These are the variables that are passed to body() and condition() in the while loop
    index_summation = (
        tf.constant(0),  # Index = 0
        tf.zeros([self.batch_size, self.num_samples]),  # w
        self.prior.sample([self.batch_size, self.num_samples]),  # z
        tf.constant(self.stepsize, shape=[self.batch_size, self.num_samples], dtype=tf.float32),
        tf.constant(self.avg_acceptance_rate, shape=[self.batch_size, self.num_samples], dtype=tf.float32),
    )

    def condition(index, w, z, stepsize, avg_acceptance_rate):
      "The condition keeps the while loop going until we reach the end of the schedule."
      return tf.less(index, len(schedule) - 1)

    def body(index, w, z, stepsize, avg_acceptance_rate):
      "The body of the while loop over the schedule."
      #
      # Get the pair of temperatures for this transition
      temp_pair = tf.gather(temp_pairs, index)
      t0 = tf.gather(temp_pair, 1)  # First temperature
      t1 = tf.gather(temp_pair, 2)  # Second temperature
      #
      # Calculate u at the new temperature and at the old one
      new_u = self.log_f_i(z, t1)
      prev_u = self.log_f_i(z, t0)
      #
      # Add the difference in u to the weight
      w = tf.add(w, new_u - prev_u)
      #
      # New step: make a HMC move
      accept, final_pos, final_vel = hmc_move(
          z,
          lambda z: self.energy_fn(z, t1),
          stepsize,
          self.n_steps
      )
      #
      # Make updates to state
      new_z, new_stepsize, new_acceptance_rate = hmc_updates(
          z,
          stepsize,
          avg_acceptance_rate=avg_acceptance_rate,
          final_pos=final_pos,
          accept=accept,
          stepsize_min=self.stepsize_min,
          stepsize_max=self.stepsize_max,
          stepsize_dec=self.stepsize_dec,
          stepsize_inc=self.stepsize_inc,
          target_acceptance_rate=self.target_acceptance_rate,
          avg_acceptance_slowness=self.avg_acceptance_slowness,
          batch_size=self.batch_size
      )
      return tf.add(index, 1), w, new_z, new_stepsize, new_acceptance_rate

    #
    # While loop across temperature schedule
    _, w, z_i, step_size, avg_acceptance_rate = \
        tf.while_loop(condition, body, index_summation, parallel_iterations=1, swap_memory=True)
    #
    # Return the log of the mean (axis=1 for samples) of exp(weights)
    with tf.control_dependencies([w, step_size, avg_acceptance_rate]):
      log_normalizer = tf.reduce_logsumexp(w, axis=1) - tf.log(tf.constant(self.num_samples, dtype=tf.float32))
      return log_normalizer, z_i, step_size, avg_acceptance_rate
