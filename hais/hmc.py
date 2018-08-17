"""
Code to implement Hamiltonian Monte Carlo.
"""

import tensorflow as tf


def tf_expand_rank(input_, rank):
  "Expand the `input_` tensor to the given rank by appending dimensions"
  while len(input_.shape) < rank:
    input_ = tf.expand_dims(input_, axis=-1)
  return input_


def tf_expand_tile(input_, to_match):
  "Expand and tile the `input_` tensor to match the `to_match` tensor."
  assert len(input_.shape) <= len(to_match.shape)
  input_rank = len(input_.shape)
  match_rank = len(to_match.shape)
  tiling = [1] * input_rank + to_match.shape.as_list()[input_rank:]
  return tf.tile(tf_expand_rank(input_, match_rank), tiling)


def kinetic_energy(v, event_axes):
  """
  Calculate the kinetic energy of the system. :math:`- \\\log \\Phi(v)` in Sohl-Dickstein and Culpepper's paper.
  Not normalised by :math:`M \\log(2 \\pi) / 2`
  """
  return 0.5 * tf.reduce_sum(tf.square(v), axis=event_axes)


def hamiltonian(position, velocity, energy_fn, event_axes):
  """
  Calculate the Hamiltonian of the system. Eqn 20 and 21 in Sohl-Dickstein and Culpepper's paper.
  """
  potential = energy_fn(position)
  momentum = kinetic_energy(velocity, event_axes)
  return potential + momentum


def metropolis_hastings_accept(E0, E1):
  """
  Accept or reject a move based on the energies of the two states.
  """
  ediff = E0 - E1
  return ediff >= tf.log(tf.random_uniform(shape=tf.shape(ediff)))


def leapfrog(x0, v0, eps, energy_fn):
  """
  Simulate the Hamiltonian dynamics using leapfrog method. That is follow the 2nd step in the 5 step
  procedure in Section 2.3 of Sohl-Dickstein and Culpepper's paper.
  """
  eps = .2
  epshalf = eps / 2.
  xhalf = x0 + epshalf * v0
  dE_dx = tf.gradients(tf.reduce_sum(energy_fn(xhalf)), xhalf)[0]
  v1 = v0 - eps * dE_dx
  x1 = xhalf + epshalf * v1
  return x1, v1


def hmc_move(x0, energy_fn, event_axes, eps):
  """
  Make a HMC move.
  """
  #
  # Choose an initial velocity randomly.
  # This does not use Sohl-Dickstein and Culpepper's idea of conserving
  # momentum across iterations.
  # TODO: fix this!
  v0 = tf.random_normal(tf.shape(x0))
  #
  # STEP 2:
  # Simulate the dynamics of the system using leapfrog
  x1, v1 = leapfrog(
      x0=x0,
      v0=v0,
      eps=eps,
      energy_fn=energy_fn
  )
  #
  # STEP 3:
  # Accept or reject according to MH
  E0 = hamiltonian(x0, v0, energy_fn, event_axes)
  E1 = hamiltonian(x1, v1, energy_fn, event_axes)
  accept = metropolis_hastings_accept(E0=E0, E1=E1)
  return accept, x1, v1


def hmc_updates(x0, eps, smoothed_acceptance_rate, x1, accept,
                target_acceptance_rate, stepsize_inc, stepsize_dec,
                stepsize_min, stepsize_max, acceptance_decay):
  """
  Do HMC updates, that is

    - update the position according to whether the move was accepted or rejected
    - update the step size adaptively according to whether the acceptance rate is high or low
    - smooth the acceptance rates (over the iterations)
  """
  #
  # Choose the new position according to whether we accepted or rejected
  # First expand the accept (which has batch shape) to full (batch + event) shape.
  print('accept: {}'.format(accept.shape))
  print('x0: {}'.format(x0.shape))
  print('x1: {}'.format(x1.shape))
  new_x = tf.where(tf_expand_tile(accept, x1), x1, x0)
  print('new_x: {}'.format(new_x.shape))
  #
  # Increase or decrease step size according to whether we are above or below target (average) acceptance rate
  # print('smoothed_acceptance_rate: {}'.format(smoothed_acceptance_rate.shape))
  # print('stepsize_inc: {}'.format(stepsize_inc.shape))
  # print('stepsize_dec: {}'.format(stepsize_dec.shape))
  new_eps = tf.where(
      smoothed_acceptance_rate > target_acceptance_rate,
      stepsize_inc,
      stepsize_dec) * eps
  #
  # Make sure we stay within specified step size range
  new_eps = tf.clip_by_value(new_eps, clip_value_min=stepsize_min, clip_value_max=stepsize_max)
  #
  # Smooth the acceptance rate
  new_acceptance_rate = tf.add(acceptance_decay * smoothed_acceptance_rate,
                               (1.0 - acceptance_decay) * tf.to_float(accept))
  return new_x, new_eps, new_acceptance_rate
