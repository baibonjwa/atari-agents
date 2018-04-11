import numpy as np
import tensorflow as tf
import _pickle as cPickle

# https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])