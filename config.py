

import os
import random
import sys

import numpy as np
import tensorflow as tf


class Real:
    def __init__(self, precision):
        self.precision = None
        self.reals = None
        if precision == 16:
            self.set_float16()
        elif precision == 32:
            self.set_float32()
        elif precision == 64:
            self.set_float64()

    def __call__(self, package):
        return self.reals[package]
    
    def set_float16(self):
        self.precision = 16
        self.reals = {np: np.float16, tf: tf.float16}

    def set_float32(self):
        self.precision = 32
        self.reals = {np: np.float32, tf: tf.float32}

    def set_float64(self):
        self.precision = 64
        self.reals = {np: np.float64, tf: tf.float64}


# default float type
real = Real(32)

def default_float():
    """Returns the default float type, as a string."""
    if real.precision == 64:
        return "float64"
    elif real.precision == 32:
        return "float32"
    elif real.precision == 16:
        return "float16"


def set_default_float(value):
    if value == "float16":
        print("Set the default float type to float16")
        real.set_float16()
    elif value == "float32":
        print("Set the default float type to float32")
        real.set_float32()
    elif value == "float64":
        print("Set the default float type to float64")
        real.set_float64()
    else:
        raise ValueError(f"{value} not supported")
    
    tf.keras.backend.set_floatx(value)

def set_random_seed(seed):
    """Sets all random seeds for the program (Python random, NumPy, and backend), and
    configures the program to run deterministically.

    You can use this to make the program fully deterministic. This means that if the
    program is run multiple times with the same inputs on the same hardware, it will
    have the exact same outputs each time. This is useful for debugging models, and for
    obtaining fully reproducible results.

    - For backend TensorFlow 2.x: Results might change if you run the model several
      times in the same terminal.

    Warning:
        Note that determinism in general comes at the expense of lower performance and
        so your model may run slower when determinism is enabled.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.random.set_seed(seed)

    global random_seed
    random_seed = seed
