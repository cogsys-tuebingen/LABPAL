import tensorflow as tf
import tables
import os


def _get_norm_of_multiple_tensors(tensors):
    norm = 0
    for tensor in tensors:
        norm += tf.reduce_sum(tensor * tensor)
    return tf.sqrt(norm)


def _get_directional_derivative(gradient_tensors, direction_tensors, direction_norm=None):
    """
    :param gradient_tensors:
    :param direction_tensors:
    :param direction_norm:
    :return: the directional derivative. Note that for PAL we need the negative directional derivative
    """
    if direction_norm is None:
        direction_norm = _get_norm_of_multiple_tensors(direction_tensors)
    scalar_product = 0
    for gradient, direction in zip(gradient_tensors, direction_tensors):
        scalar_product += tf.reduce_sum(gradient * direction)
    directional_derivative = scalar_product / direction_norm
    return directional_derivative


def _get_angle_of_multiple_tensors(tensors1, tensors2):
    norm1 = 0
    norm2 = 0
    scalar_product = 0
    for tensor1, tensor2 in zip(tensors1, tensors2):
        norm1 += tf.reduce_sum(tensor1 * tensor1)
        norm2 += tf.reduce_sum(tensor2 * tensor2)
        scalar_product += tf.reduce_sum(tensor1 * tensor2)
    norm1 = tf.sqrt(norm1)
    norm2 = tf.sqrt(norm2)
    angle = tf.acos(scalar_product / (norm1 * norm2))
    return angle
