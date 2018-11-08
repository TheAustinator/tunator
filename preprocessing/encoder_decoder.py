import tensorflow as tf
import numpy as np


hidden_size = 512


def build_network(self):
    input_dim = (2, 4,5)
    with tf.name_scope('placeholder_inputs'):
        self.inputs = [
            tf.placeholder(
                tf.float32,
                (self.batch_size, input_dim),
                name='input',
            )
        ]

    with tf.name_scope('placeholder_targets'):
        self.targets = [
            tf.placeholder(
                tf.float32,
                (self.batch_size, input_dim),
                name='target'
            )
        ]

    with tf.name_scope('placeholder_hidden'):
        self.hidden = [
            tf.placeholder(
                tf.float32,
                (self.batch_size, hidden_size)
            )
        ]
