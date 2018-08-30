"""
Calculations of exact log marginal likelihoods for certain models.
"""


import numpy as np
from scipy import linalg as la
import scipy.stats as st

LOG_2_PI = np.log(2. * np.pi)


def culpepper1a_log_marginal(x, phi, sigma_n):
  """Calculate the exact log marginal likelihood of the `x` given
  `phi` and `sigma_n` in model 1a (Gaussian prior) from Culpepper
  and Sohl-Dickstein (2011)."""
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
  """Model 1a (Gaussian prior) from Culpepper and Sohl-Dickstein."""
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
