from __future__ import division

import os
import pdb # pylint: disable=unused-import
import functools
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim # pylint: disable=E0611
from scipy.misc import imresize
from .history import History

def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class Qnetwork():
    # pylint: disable=too-many-instance-attributes
    def __init__(self, h_size, action_space, name):
        self.w = {}
        self.q = {}
        self.t_w = {}
        with tf.variable_scope(name):
            self.input_data = tf.placeholder('float32', [None, 84, 84, 4], name="input_data")

            self.conv1 = slim.conv2d(self.input_data, 32, 8, 4, 'VALID',
                                    biases_initializer=None, scope='conv1')
            self.conv2 = slim.conv2d(self.conv1, 64, 4, 2, 'VALID',
                                    biases_initializer=None, scope='conv2')
            self.conv3 = slim.conv2d(self.conv2, 64, 3, 1, 'VALID',
                                    biases_initializer=None, scope='conv3')

            shape = self.conv3.get_shape().as_list()
            self.conv3_flat = tf.reshape(self.conv3,
                                        [-1, functools.reduce(lambda x, y: x * y, shape[1:])])

            self.conv4 = slim.fully_connected(self.conv3_flat, 512, activation_fn=tf.nn.relu, scope='conv4')
            self.Qout = slim.fully_connected(self.conv4, action_space, activation_fn=None, scope='Qout')

            # Dueling
            # self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
            #     linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, name=name + 'value_hid')

            # self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
            #     linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, name=name + 'adv_hid')

            # self.Value, self.w['val_w_out'], self.w['val_w_b'] = \
            # linear(self.value_hid, 1, name=name + 'value_out')

            # self.Advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
            # linear(self.adv_hid, action_space, name=name + 'adv_out')

            # self.Qout = self.Value + tf.subtract(
            #     self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

            self.predict = tf.argmax(self.Qout, 1)

class DoubleDuelingDQNAgent(object):
    def __init__(self, env, sess, FLAGS):

        self.env = env
        self.history = History()
        self.action_space = env.action_space

        self.config = {
            #  "batch_size": 16,
            "batch_size": 32,
            # "update_freq": 4,
            "update_freq": 4,
            # "update_freq": 500,
            "y": .99,
            "startE": 1.0,
            "endE": 0.1,
            "total_steps": 2500000,
            # "annealing_steps": 10000,
            "annealing_steps": 50000,
            "num_episodes": 10000,
            "pre_train_steps": 2500,
            # "pre_train_steps": 2,
            "max_epLength": 1000,
            "screen_width": 84,
            "screen_height": 84,
            "load_model": False,
            "path": "./ckpt",
            "h_size": 512,
            "tau": 0.001,
            "target_q_update_step": 500,
        }

        self.mainQN = Qnetwork(self.config["h_size"], env.action_space.n, 'main')
        self.targetQN = Qnetwork(self.config["h_size"], env.action_space.n, 'target')
        self.sess = sess

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
            self.actions_onehot = tf.one_hot(self.actions, env.action_space.n, dtype=tf.float32, name="action_onehot")
            # self.action = tf.placeholder(shape=[None], dtype=tf.int32, name= "action")
            self.q = tf.reduce_sum(tf.multiply(self.mainQN.Qout, self.actions_onehot), axis=1)
            # self.td_error = tf.square(self.target_q_t - self.q)
            self.td_error = self.target_q_t - self.q
            self.loss = tf.reduce_mean(clipped_error(self.td_error))

            self.learning_rate = 0.00025
            self.learning_rate_minimum = 0.00025
            self.learning_rate_decay = 0.96
            self.learning_rate_decay_step = 5 * 100

            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                tf.train.exponential_decay(
                    self.learning_rate,
                    self.learning_rate_step,
                    self.learning_rate_decay_step,
                    self.learning_rate_decay,
                    staircase=True))
            # self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
            # self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
            # self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
            # self.trainer = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)
            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['avg.reward', 'avg.loss', 'avg.q', \
                'ep.reward.max', 'ep.reward.min', 'ep.reward.avg', 'ep.num_of_game', 'training.learning_rate', 'e']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag]  = tf.summary.scalar("%s" % (tag), self.summary_placeholders[tag])

            histogram_summary_tags = ['ep.rewards', 'ep.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

        self.saver = tf.train.Saver()
        self.trainables = tf.trainable_variables()
        self.targetOps = self.updateTargetGraph(self.trainables, self.config["tau"])

        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.e = self.config["startE"]
        self.stepDrop = (self.config["startE"] - self.config["endE"]) \
            / self.config["annealing_steps"]

        self.jList = []
        self.rList = []
        self.update_count = 1
        self.total_loss = 0.
        self.total_q = 0.

        if not os.path.exists(self.config["path"]):
            os.makedirs(self.config["path"])

        log_path = "%s/%s/%s/%s" % (FLAGS.log_dir,
                                    FLAGS.env_name,
                                    str(self.__class__.__name__),
                                    FLAGS.timestamp)
        self.writer = tf.summary.FileWriter("%s/%s" % (log_path, '/train'), sess.graph)
        tf.train.write_graph(self.sess.graph, './', 'dqn.pb', False)
        tf.train.write_graph(self.sess.graph, './', 'dqn.pbtxt')

    def learn(self, step_i, state, reward, action, done):

        self.history.add(state)
        self.memory.add(state, reward, action, done)

        loss = .0
        if step_i > self.config["pre_train_steps"]:
            if self.memory.count < 4:
               return
            if self.e > self.config["endE"]:
                self.e -= self.stepDrop
            if step_i % (self.config["update_freq"]) == 0:
                s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
                # trainBatch = self.memory.sample(self.config["batch_size"])

                # Double Q
                # self.lastStates = np.stack(trainBatch[:, 3])
                # Q1 = self.sess.run(self.mainQN.predict, feed_dict={
                #     self.mainQN.input_data:np.stack(trainBatch[:, 3])
                # })
                # Q2 = self.sess.run(self.targetQN.Qout, feed_dict={
                #     self.targetQN.input_data:np.stack(trainBatch[:, 3])
                # })
                # end_multiplier = -(trainBatch[:, 4] - 1)
                # doubleQ = Q2[range(self.config["batch_size"]), Q1]
                # targetQ = trainBatch[:, 2] + (self.config["y"] * doubleQ * end_multiplier)

                # _, loss = self.sess.run(
                #     [self.optimizer, self.loss],
                #     feed_dict={
                #         self.mainQN.input_data:np.stack(trainBatch[:, 0]),
                #         self.targetQ:targetQ,
                #         self.actions:trainBatch[:, 1],
                #     })

                q_t_plus_1 = self.sess.run(self.targetQN.Qout, feed_dict={ self.targetQN.input_data:s_t_plus_1 })
                terminal = np.array(terminal) + 0.
                max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
                target_q_t = (1. - terminal) * 0.99 * max_q_t_plus_1 + reward

                _, q_t, loss = self.sess.run(
                    [self.optimizer, self.mainQN.Qout, self.loss],
                    feed_dict={
                        self.target_q_t: target_q_t,
                        self.actions: action,
                        self.mainQN.input_data:s_t,
                        self.learning_rate_step: step_i,
                    })
                self.total_loss += loss
                self.total_q += q_t.mean()
                self.update_count += 1


            if step_i % 500 == 499:
                self.updateTarget(self.targetOps, self.sess)

        return self.total_loss, self.total_q, self.update_count, state, loss, self.e

    def act(self, step_i, env):
        if np.random.rand(1) < self.e or step_i < self.config["pre_train_steps"]:
            a = np.random.randint(0, self.env.action_space.n)
        else:
            a = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.input_data:[self.history.get()]})[0]
        # use env rather than self.env because self.env is Gym object and env is Environemnt object
        obs, reward, done, _ = env.step(a)
        self.env.render()
        return a, obs, reward, done, _

    def updateTargetGraph(self, tfVars, tau):
        with tf.variable_scope('update_target_graph'):
            total_vars = len(tfVars)
            op_holder = []
            for idx, var in enumerate(tfVars[0:total_vars//2]):
                op_holder.append(tfVars[idx + total_vars//2].assign(var.value()))
                # tfVars[idx + total_vars//2].assign(
                #     (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars//2].value())))
            return op_holder

    def updateTarget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)