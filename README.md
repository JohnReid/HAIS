# README

A TensorFlow implementation of Hamiltonian Annealed Importance Sampling (HAIS).
We implement the [method](http://arxiv.org/abs/1205.1925) described by Sohl-Dickstein and Culpepper
in their paper "Hamiltonian Annealed Importance Sampling for partition function estimation".


### Features

  - Partial momentum refresh (from HAIS paper). This preserves some fraction of the Hamiltonian Monte
    Carlo momentum across annealing distributions resulting in more accurate estimation.
  - Adaptive step size for Hamiltonian Monte Carlo. This is a simple scheme to adjust the step size for
    each chain in order to push the smoothed acceptance rate towards a theoretical optimum.


### Related implementations

We have used ideas and built upon the code from some of the following repositories:

  - BayesFlow TensorFlow 1.4 - 1.6 [contribution](https://www.tensorflow.org/versions/r1.6/api_docs/python/tf/contrib/bayesflow/hmc/ais_chain).
    This is now integrated into [TensorFlow Probability](https://github.com/tensorflow/probability).
  - Sohl-Dickstein's Matlab [implementation](https://github.com/Sohl-Dickstein/Hamiltonian-Annealed-Importance-Sampling)
  - Xuechen Li's PyTorch (0.2.0) [implementation](https://github.com/lxuechen/BDMC) of Bi-Directional Monte Carlo
    from ["Sandwiching the marginal likelihood using bidirectional Monte Carlo"](https://arxiv.org/abs/1511.02543)
  - Tony Wu's Theano/Lasagne [implementation](https://github.com/tonywu95/eval_gen) of the methods described in
    ["On the Quantitative Analysis of Decoder-Based Generative Models"](https://arxiv.org/abs/1611.04273)
  - jiamings's (unfinished?) TensorFlow [implementation](https://github.com/jiamings/ais/) based on Tony Wu's Theano code.
  - Stefan Webb's HMC AIS in tensorflow [repository](https://github.com/stefanwebb/tensorflow-hmc-ais).


### Tests

The tests that appear to be working include:

  - `test-hmc`: a simple test of the HMC implementation
  - `test-hmc-mvn`: a test of the HMC implementation that samples from a multivariate normal
  - `test-hais-log-gamma`: a simple test to sample from and calculate the log normaliser of
    an unnormalised log-Gamma density.
  - `test-hais-model1a-gaussian`: a test that estimates the log marginal likelihood for model 1a with
    a Gaussian prior from Sohl-Dickstein and Culpepper (2011).


### Installation

Install either the GPU version of TensorFlow (I don't know why but `tensorflow-gpu==1.8` and
`tensorflow-gpu==1.9` are >10x slower than 1.7 on my machine)
```bash
pip install tensorflow-gpu==1.7
```
or the CPU version
```bash
pip install tensorflow
```
then install the project
```bash
pip install git+https://github.com/JohnReid/HAIS
```


### API documentation

The implementation contains some [documentation](https://johnreid.github.io/HAIS/build/html/index.html)
generated from the docstrings that may be useful. However it is probably easier to examine the
[test scripts](https://github.com/JohnReid/HAIS/tree/master/tests) and adapt them to your needs.


### Who do I talk to?

[John Reid](https://twitter.com/__Reidy__) or [Halil Bilgin](https://twitter.com/bilginhalil)
