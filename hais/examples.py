"""
Unnormalised targets and exact calculations for some example problems.

  - An unnormalised log-Gamma distribution
  - Model 1a from Sohl-Dickstein and Culpepper

"""


import numpy as np
import scipy.linalg as la
import scipy.special as sp
import scipy.stats as st
import tensorflow as tf
tfd = tf.contrib.distributions


LOG_2_PI = np.log(2. * np.pi)


def log_gamma_unnormalised_lpdf(x, alpha, beta):
  """
  Unnormalized log probability density function of the log-gamma(ALPHA, BETA) distribution.

  The probability density function for a gamma distribution is:

  .. math::

      f(x; \\alpha, \\beta) =
        \\frac{\\beta^\\alpha}{\Gamma(\\alpha)}
        x^{\\alpha-1}
        e^{- \\beta x}

  for all :math:`x > 0` and any given shape :math:`\\alpha > 0` and rate :math:`\\beta > 0`. Given a change
  of variables :math:`y = \\log(x)` we have the density for a log-gamma distribution:

  .. math::

      f(y; \\alpha, \\beta) =
        \\frac{\\beta^\\alpha}{\Gamma(\\alpha)}
        e^{\\alpha y - \\beta e^y}

  """
  return alpha * x - beta * tf.exp(x)


def log_gamma_exact_log_normaliser(alpha, beta):
  """The exact log normalizer is:

  .. math::

    \\log \\Gamma(\\alpha) - \\alpha \\log \\beta
  """
  return sp.gammaln(alpha) - alpha * np.log(beta)


class Culpepper1aGaussian(object):
  """Implementations of likelihood, sampling and exact marginal
  for model1a (with Gaussian prior) from Sohl-Dickstein and
  Culpepper."""

  def __init__(self, M, L, batch_size, sigma_n):
    """Initialise the model with the parameters."""
    self.M = M
    self.L = L
    self.batch_size = batch_size
    self.sigma_n = sigma_n


  def sample(self):
    """Sample from model."""
    #
    # Sample phi
    phi = st.norm.rvs(size=(self.M, self.L)).astype(dtype=np.float32)
    #
    # Sample z
    z = st.norm.rvs(size=(self.batch_size, self.L)).astype(dtype=np.float32)
    #
    # Sample x
    x_loc = (phi@z.T).T
    px = st.norm(loc=x_loc, scale=self.sigma_n)
    x = px.rvs(size=(self.batch_size, self.M))
    #
    return phi, z, x_loc, px, x


  def prior(self, n_chains):
    """Returns a prior distribution for the model."""
    return tfd.MultivariateNormalDiag(loc=tf.zeros([self.batch_size, n_chains, self.L]))


  def log_likelihood_fn(self, x_ph, tf_phi, n_chains):
    "Returns a function to calculate the log pdf of the conditional distribution of x given z."
    #
    def ll_fn(z):
      assert (self.batch_size, n_chains, self.L) == z.shape
      assert (self.M, self.L) == tf_phi.shape
      assert (self.batch_size, self.M) == x_ph.shape
      loc = tf.squeeze(
          tf.matmul(
              tf.tile(tf.expand_dims(tf.expand_dims(tf_phi, axis=0), axis=0), [self.batch_size, n_chains, 1, 1]),
              tf.expand_dims(z, axis=-1)),
          axis=-1)
      assert (self.batch_size, n_chains, self.M) == loc.shape
      x_given_z = tfd.MultivariateNormalDiag(loc=tf.cast(loc, tf.float32), scale_identity_multiplier=self.sigma_n)
      return x_given_z.log_prob(
          tf.tile(tf.expand_dims(x_ph, axis=1), [1, n_chains, 1]), name='log_likelihood')
    #
    return ll_fn


  def log_marginal(self, x, phi):
    """Calculate the exact log marginal likelihood of the `x` given
    `phi` and `sigma_n`."""
    #
    # Predictive covariance of x is sum of covariance of phi a and covariance of x|a
    x_Sigma = phi@phi.T + np.diag(self.sigma_n**2 * np.ones(self.M))
    #
    # Predictive mean is 0 by symmetry
    # so given that x is distributed as a MVN, the exact marginal is
    lp_exact = st.multivariate_normal.logpdf(x, cov=x_Sigma)
    #
    return lp_exact


def _culpepper1a_log_marginal_overcomplicated(x, phi, sigma_n):
  """An over-complicated and incorrect method to calculate
  the exact marginal likelihood for model 1a (Gaussian prior) from Sohl-Dickstein and Culpepper."""
  raise NotImplementedError('This is an overcomplicated implementation that does not work')
  M, L = phi.shape
  sigma_n2 = sigma_n**2
  #
  # Precision of posterior for a
  SigmaInv = np.diag(np.ones(L)) + phi.T @ phi / sigma_n2
  #
  # Cholesky
  C = la.cholesky(SigmaInv)
  halflogSigmaDet = - np.add.reduce(np.log(np.diag(C)))
  #
  # Solve for term we need
  xPhiCinv = la.solve_triangular(C, phi.T @ x.T, lower=True).T
  #
  # Normalising constants
  lZa = L / 2. * LOG_2_PI
  lZxa = M / 2. * LOG_2_PI + M * np.log(sigma_n)
  lZax = L / 2. * LOG_2_PI + halflogSigmaDet
  #
  # Log marginal
  lpx = - lZa - lZxa + lZax + (np.square(xPhiCinv).sum(axis=1) / sigma_n2 - np.square(x).sum(axis=1)) / (2. * sigma_n2)
  #
  return lpx
