"""Hamiltonian Annealed Importance Sampling (HAIS)

Sohl-Dickstein and Culpepper "Hamiltonian Annealed Importance Sampling for partition function
estimation" (2011).
"""

from packaging import version
from hais.ais import HAIS, get_schedule  # noqa: F401
import tensorflow as tf

#
# Configure TensorFlow depending on version
if version.parse(tf.__version__) >= version.parse('2.0.0'):
    # TensorFlow version 2
    # Using TFv1 compatibility mode in TF2
    tf = tf.compat.v1
