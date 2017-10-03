"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import randint
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets('data/', one_hot=True)

rnd = randint(0, 10000)
for i in range(0, 784):
    if i % 28 == 0:
        print("\n", end='')
    if mnist.test.images[rnd][i] > 0.5:
        print("@", end='')
    else:
        print(" ", end='')
    if i == 783:
        print("\n")

#  def main(_):
  # Import data

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#  y = tf.matmul(x, W) + b
y = tf.nn.softmax(tf.matmul(x, W) + b)


# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
#  cross_entropy = tf.reduce_mean(
    #  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


rnd = randint(0, 10000)

for i in range(0, 784):
    if i % 28 == 0:
        print("\n", end='')
    if mnist.test.images[rnd][i] > 0.5:
        print("@", end='')
    else:
        print(" ", end='')
    if i == 783:
        print("\n")

yy = tf.argmax(y, 1)
yy_ = tf.argmax(y_, 1)

rx = mnist.test.images[rnd:rnd + 1]
ry_ = mnist.test.labels[rnd:rnd + 1]

print(sess.run(yy, feed_dict={x: rx, y_: ry_}))
print(sess.run(yy_, feed_dict={x: rx, y_: ry_}))

print(sess.run(y, feed_dict={x: rx, y_: ry_}))
print(sess.run(y_, feed_dict={x: rx, y_: ry_}))

#  print(sess.run(correct_prediction, feed_dict={x: rx,
    #  y_: ry_}));


#  if __name__ == '__main__':
  #  parser = argparse.ArgumentParser()
  #  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      #  help='Directory for storing input data')
  #  FLAGS, unparsed = parser.parse_known_args()
  #  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
