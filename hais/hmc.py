"""
Code to implement Hamiltonian Monte Carlo.
"""

import tensorflow as tf


def kinetic_energy(v):
  """
  Calculate the kinetic energy of the system.
  """
  return 0.5 * tf.reduce_sum(tf.multiply(v, v), axis=2)


def hamiltonian(position, velocity, energy_fn):
  """
  Calculate the Hamiltonian of the system.
  """
  potential = energy_fn(position)
  momentum = kinetic_energy(velocity)
  return potential + momentum


def metropolis_hastings_accept(energy_prev, energy_next):
  """
  Accept or reject a move based on the energy.
  """
  ediff = energy_prev - energy_next
  accept = ediff >= tf.log(tf.random_uniform(tf.shape(energy_prev)))
  return accept


def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
  """
  Simulate the Hamiltonian dynamics using leapfrop method.
  """

  def leapfrog(pos, vel, step, i, dE_dpos):
    "Make a leapfrog step."
    energy = energy_fn(pos)
    dE_dpos = tf.gradients(tf.reduce_sum(energy), pos)[0]

    new_vel = vel - step * dE_dpos
    new_pos = pos + step * new_vel
    return new_pos, new_vel, step, tf.add(i, 1), dE_dpos

  def condition(pos, vel, step, i, dE_dpos):
    "Condition to stop after so many steps."
    return tf.less(i, n_steps)

  #
  # Set up variables to enter while loop
  dE_dpos = tf.cast(tf.gradients(tf.reduce_sum(energy_fn(initial_pos)), initial_pos)[0], tf.float32)
  # A stepsize variable expanded for each position
  stepsize_pos = tf.expand_dims(stepsize, axis=-1)
  vel_half_step = initial_vel - 0.5 * stepsize_pos * dE_dpos
  pos_full_step = initial_pos + stepsize_pos * vel_half_step
  #
  # Run while loop
  final_pos, new_vel, _, _, _ = tf.while_loop(
      condition,
      leapfrog,
      [pos_full_step, vel_half_step, stepsize_pos, tf.constant(0), dE_dpos])
  dE_dpos = tf.gradients(tf.reduce_sum(energy_fn(final_pos)), final_pos)[0]
  final_vel = new_vel - 0.5 * stepsize_pos * dE_dpos
  return final_pos, final_vel


def hmc_move(initial_pos, energy_fn, stepsize, n_steps):
  """
  Make a HMC move.
  """
  #
  # Choose an initial velocity randomly
  initial_vel = tf.random_normal(tf.shape(initial_pos))
  #
  # Simulate the dynamics of the system
  final_pos, final_vel = simulate_dynamics(
      initial_pos=initial_pos,
      initial_vel=initial_vel,
      stepsize=stepsize,
      n_steps=n_steps,
      energy_fn=energy_fn
  )
  #
  # Accept or reject according to MH
  energy_prev = hamiltonian(initial_pos, initial_vel, energy_fn)
  energy_next = hamiltonian(final_pos, final_vel, energy_fn)
  accept = metropolis_hastings_accept(energy_prev=energy_prev, energy_next=energy_next)
  # tf.summary.scalar('accept', tf.reduce_mean(tf.cast(accept, tf.float32)))
  return accept, final_pos, final_vel


def hmc_updates(initial_pos, stepsize, avg_acceptance_rate, final_pos, accept,
                target_acceptance_rate, stepsize_inc, stepsize_dec,
                stepsize_min, stepsize_max, avg_acceptance_slowness, batch_size):
  """
  Do HMC updates, that is

    - update the position according to whether the move was accepted or rejected
    - update the step size adaptively according to whether the acceptance rate is high or low
    - smoothly estimate the acceptance rates (over the iterations)
  """
  #
  # Choose the new position according to whether we accepted or rejected
  new_pos = tf.where(tf.tile(tf.expand_dims(accept, axis=-1), [1, 1, final_pos.shape[-1]]), final_pos, initial_pos)
  #
  # Increase or decrease step size according to whether we are above or below target (average) acceptance rate
  new_stepsize_ = tf.where(
      avg_acceptance_rate > target_acceptance_rate,
      stepsize_inc,
      stepsize_dec) * stepsize
  #
  # Make sure we stay within specified step size range
  new_stepsize = tf.clip_by_value(new_stepsize_, clip_value_min=stepsize_min, clip_value_max=stepsize_max)
  #
  # Update the acceptance rate?
  new_acceptance_rate = tf.add(avg_acceptance_slowness * avg_acceptance_rate,
                               (1.0 - avg_acceptance_slowness) * tf.to_float(accept))
  return new_pos, new_stepsize, new_acceptance_rate
