.. HAIS documentation master file, created by
   sphinx-quickstart on Fri Aug 31 10:19:01 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

HAIS
====

This is a tensorflow implementation of Hamiltonian Annealed Importance Sampling
(HAIS). HAIS is a technique to estimate partition functions (normalising
constants of probability densities) and sample from such densities. HAIS was
proposed by Sohl-Dickstein and Culpepper (2011).

Annealed Importance Sampling (AIS) is a technique to sample from unnormalised
complex densities with isolated modes that also can be used to estimate
normalising constants. It combines importance sampling with Markov chain
sampling methods. The canonical reference is "Annealed Importance Sampling"
Radford M. Neal (Technical Report No. 9805, Department of Statistics,
University of Toronto, 1998)

HAIS is a version of AIS that uses Hamiltonian Monte Carlo (as opposed to
Metropolis-Hastings or some other sampler) to move between the annealed
distributions.

Developed by `@__Reidy__ <https://twitter.com/__Reidy__>`_ and
`@bilginhalil <https://twitter.com/bilginhalil>`_
and derived from https://github.com/jiamings/ais/.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ais.rst
   hmc.rst
   examples.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
