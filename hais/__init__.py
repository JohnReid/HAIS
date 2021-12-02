"""Hamiltonian Annealed Importance Sampling (HAIS)

Sohl-Dickstein and Culpepper "Hamiltonian Annealed Importance Sampling for partition function
estimation" (2011).
"""

from packaging import version
from contextlib import contextmanager
from time import time
from hais.ais import HAIS, get_schedule  # noqa: F401
import tensorflow as tf

#
# Configure TensorFlow depending on version
if version.parse(tf.__version__) >= version.parse('2.0.0'):
    # TensorFlow version 2
    # Using TFv1 compatibility mode in TF2
    tf = tf.compat.v1


@contextmanager
def timing(description: str, verbose: bool=False) -> None:
    """A context manager that prints how long the context took to execute."""
    if verbose:
        print(f'{description}')
    start = time()
    yield
    elapsed_time = time() - start
    print(f'{description} took {elapsed_time:.3f}s')
