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


def culpepper1a_log_marginal(x, phi, sigma_n):
  """Calculate the exact log marginal likelihood of the `x` given
  `phi` and `sigma_n` in model 1a (Gaussian prior) from
  Sohl-Dickstein and Culpepper (2011)."""
  M, L = phi.shape
  #
  # Predictive covariance of x is sum of covariance of phi a and covariance of x|a
  x_Sigma = phi@phi.T + np.diag(sigma_n**2 * np.ones(M))
  #
  # Predictive mean is 0 by symmetry
  # so given that x is distributed as a MVN, the exact marginal is
  lp_exact = st.multivariate_normal.logpdf(x, cov=x_Sigma)
  #
  return lp_exact


def culpepper1a_log_marginal_overcomplicated(x, phi, sigma_n):
  """An over-complicated and probably incorrect method to calculate
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
