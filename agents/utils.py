import numpy as np
import tensorflow as tf
import _pickle as cPickle

# https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def variable_summaries(var, name, last=100):
    with tf.name_scope('summaries.%s' % (name)):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('%s.avg' % (name), mean)
        tf.summary.scalar('%s.stddev' % (name), tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
        tf.summary.scalar('%s.max' % (name), tf.reduce_max(var))
        tf.summary.scalar('%s.min' % (name), tf.reduce_min(var))
        tf.summary.histogram('%s.histogram' % (name), var)
    with tf.name_scope('summaries.%s.last.%s' % (name, last)):
        var_last = var[-last:]
        tf.summary.scalar('%s.avg' % (name), mean)
        tf.summary.scalar('%s.stddev' % (name), tf.sqrt(tf.reduce_mean(tf.square(var_last - mean))))
        tf.summary.scalar('%s.max' % (name), tf.reduce_max(var_last))
        tf.summary.scalar('%s.min' % (name), tf.reduce_min(var_last))
        tf.summary.histogram('%s.histogram' % (name), var_last)