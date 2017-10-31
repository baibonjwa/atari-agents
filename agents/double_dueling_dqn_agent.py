from __future__ import division

import os
import random
import pdb
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from .utils import rgb2gray

class Qnetwork():
    def __init__(self, h_size, action_space):
        self.scalarInput = tf.placeholder(shape=[None, 33600], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 210, 160])
        self.conv1 = slim.conv2d(
            inputs=self.imageIn,
            num_outputs=32,
            kernel_size=8,
            stride=7,
            padding='VALID',
            biases_initializer=None
        )
        self.conv2 = slim.conv2d(
            inputs=self.conv1,
            num_outputs=64,
            kernel_size=4,
            stride=3,
            padding='VALID',
            biases_initializer=None
        )
        self.conv3 = slim.conv2d(
            inputs=self.conv2,
            num_outputs=64,
            kernel_size=3,
            stride=2,
            padding='VALID',
            biases_initializer=None
        )

        #  self.conv4 = slim.conv2d(
            #  inputs=self.conv3,
            #  num_outputs=64,
            #  kernel_size=7,
            #  stride=1,
            #  padding='VALID',
            #  biases_initializer=None
        #  )


        #  self.fc = slim.fully_connected(
            #  self.conv3,
            #  num_outputs=23,
            #  activation_fn=tf.tanh,
        #  )

        #  self.conv4 = slim.conv2d(
            #  inputs=self.conv3,
            #  num_outputs=h_size,
            #  kernel_size=7,
            #  stride=1,
            #  padding='VALID',
            #  biases_initializer=None
        #  )

        # Dueling network
        #  self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        #  pdb.set_trace()
        self.streamAC, self.streamVC = tf.split(self.conv3, 2)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2, action_space]))
        self.VW = tf.Variable(xavier_init([h_size//2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.AW)

        # ?
        pdb.set_trace()
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_space, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) -  self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


class DoubleDuelingDQNAgent(object):
    def __init__(self, env, sess, FLAGS):

        self.env = env
        self.action_space = env.action_space

        self.config = {
            "batch_size": 32,
            #  "batch_size": 8,
            "update_freq": 4,
            "y": .99,
            "startE": 1,
            "endE": 0.1,
            "annealing_steps": 10000,
            "num_episodes": 10000,
            "pre_train_steps": 10000,
            "max_epLength": 50,
            "load_model": False,
            "path": "./dqn",
            "h_size": 512,
            "tau": 0.001,
        }

        #  tf.reset_default_graph()

        self.mainQN = Qnetwork(self.config["h_size"], env.action_space.n)
        self.targetQN = Qnetwork(self.config["h_size"], env.action_space.n)

        init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()
        self.trainables = tf.trainable_variables()
        self.targetOps = self.updateTargetGraph(self.trainables, self.config["tau"])

        self.sess = sess
        self.sess.run(init)
        if self.config["load_model"]:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.config["path"])
            self.saver.restore(sess, ckpt.model_checkpoint_path)

        self.e = self.config["startE"]
        self.stepDrop = (self.config["startE"] - self.config["endE"]) / self.config["annealing_steps"]

        self.jList = []
        self.rList = []
        self.total_steps = 0
        self.loss = 0

        if not os.path.exists(self.config["path"]):
            os.makedirs(self.config["path"])

        self.merged = tf.summary.merge_all()

        log_path = "%s/%s/%s/%s" % (FLAGS.log_dir, FLAGS.env_name, str(self.__class__.__name__), FLAGS.timestamp)
        self.writer = tf.summary.FileWriter("%s/%s" % (log_path, '/train'), sess.graph)

    def learn(self, state, action, reward, done, episodeBuffer, myBuffer):
        s1 = self.processState(state)
        self.total_steps += 1
        episodeBuffer.add(np.reshape(np.array([state, action, reward, s1, done]), [1, 5]))

        if self.total_steps > self.config["pre_train_steps"]:
            if self.e > self.config["endE"]:
                self.e -= self.stepDrop

            if self.total_steps % (self.config["update_freq"]) == 0:
                trainBatch = myBuffer.sample(self.config["batch_size"])
                Q1 = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput:np.vstack(trainBatch[:, 3])})
                Q2 = self.sess.run(self.targetQN.Qout, feed_dict={self.targetQN.scalarInput:np.vstack(trainBatch[:, 3])})
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(self.config["batch_size"]), Q1]
                targetQ = trainBatch[:, 2] + (self.config["y"] * doubleQ * end_multiplier)
                #  pdb.set_trace()
                _, loss = self.sess.run([self.mainQN.updateModel, self.mainQN.loss], feed_dict={ self.mainQN.scalarInput:np.vstack(trainBatch[:, 0]), self.mainQN.targetQ:targetQ, self.mainQN.actions:trainBatch[:, 1] })
                self.loss = loss
                self.updateTarget(self.targetOps, self.sess)
                #  summary = self.sess.run([self.merged], feed_dict={ self.mainQN.scalarInput:np.vstack(trainBatch[:, 0]), self.mainQN.targetQ:targetQ, self.mainQN.actions:trainBatch[:, 1] })
                #  self.train_writer.add_summary(summary, self.total_steps)
        return s1, self.loss, self.e

    def reset(self):
        self.epsilon = [0.05, 0.2]

    def act(self, obs, reward, done):
        if np.random.rand(1) < self.e or self.total_steps < self.config["pre_train_steps"]:
            a = np.random.randint(0, 4)
        else:
            a = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput:[obs]})[0]
        return a

    def processState(self, states):
        return np.reshape(states, [states.size])

    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(tfVars[idx + total_vars//2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars//2].value())))
        return op_holder

    def updateTarget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)
