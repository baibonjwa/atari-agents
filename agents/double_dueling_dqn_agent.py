from __future__ import division

import os
import pdb # pylint: disable=unused-import
import functools
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim # pylint: disable=E0611
from scipy.misc import imresize
from .utils import variable_summaries, rgb2gray
from .history import History
from .memory import Memory

def clipped_error(x):
  # Huber loss
  try:
    return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
  except:
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
  with tf.variable_scope(name):
    if data_format == 'NCHW':
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
    elif data_format == 'NHWC':
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

    w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
    conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

    b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    out = tf.nn.bias_add(conv, b, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size],
                            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            return activation_fn(out), w, b
        return out, w, b

# class Qnetwork():
#     # pylint: disable=too-many-instance-attributes
#     def __init__(self, h_size, action_space, name):
#         self.w = {}
#         self.q = {}
#         self.t_w = {}
#         with tf.variable_scope(name):
#             self.input_data = tf.placeholder('float32', [None, 84, 84, 4], name="input_data")

#             # self.conv1 = slim.conv2d(self.input_data, 32, 8, 4, 'VALID',
#             #                         biases_initializer=None, scope='conv1')
#             # self.conv2 = slim.conv2d(self.conv1, 64, 4, 2, 'VALID',
#             #                         biases_initializer=None, scope='conv2')
#             # self.conv3 = slim.conv2d(self.conv2, 64, 3, 1, 'VALID',
#             #                         biases_initializer=None, scope='conv3')

#             # shape = self.conv3.get_shape().as_list()
#             # self.conv3_flat = tf.reshape(self.conv3,
#             #                             [-1, functools.reduce(lambda x, y: x * y, shape[1:])])

#             self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.input_data,
#                 32, [8, 8], [4, 4], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='l1')
#             self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
#                 64, [4, 4], [2, 2], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='l2')
#             self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
#                 64, [3, 3], [1, 1], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='l3')

#             shape = self.l3.get_shape().as_list()
#             self.l3_flat = tf.reshape(self.l3, [-1, functools.reduce(lambda x, y: x * y, shape[1:])])

#             self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=tf.nn.relu, name='l4')
#             self.Qout, self.w['q_w'], self.w['q_b'] = linear(self.l4, action_space, name='q')

#             # self.conv4, self.w['l4_w'], self.w['l4_b'] = linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, name='conv4')
#             # self.Qout, self.q['q_w'], self.q['q_b'] = linear(self.conv4, action_space, name='Qout')

#             # Dueling
#             # self.value_hid, self.w['l4_val_w'], self.w['l4_val_b'] = \
#             #     linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, name=name + 'value_hid')

#             # self.adv_hid, self.w['l4_adv_w'], self.w['l4_adv_b'] = \
#             #     linear(self.conv3_flat, 512, activation_fn=tf.nn.relu, name=name + 'adv_hid')

#             # self.Value, self.w['val_w_out'], self.w['val_w_b'] = \
#             # linear(self.value_hid, 1, name=name + 'value_out')

#             # self.Advantage, self.w['adv_w_out'], self.w['adv_w_b'] = \
#             # linear(self.adv_hid, action_space, name=name + 'adv_out')

#             # self.Qout = self.Value + tf.subtract(
#             #     self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))

#             self.predict = tf.argmax(self.Qout, 1)

#         # optimizer
#         # self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
#         # self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
#         # self.actions_onehot = tf.one_hot(self.actions, action_space, dtype=tf.float32)

#         # self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

#         # self.td_error = tf.square(self.targetQ - self.Q)
#         # self.loss = tf.reduce_mean(self.td_error)
#         # # self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
#         # # self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
#         # # self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
#         # self.trainer = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)
#         # self.optimizer = self.trainer.minimize(self.loss)

class DoubleDuelingDQNAgent(object):
    def __init__(self, env, sess, FLAGS):

        self.env = env
        self.history = History()
        self.memory = Memory()
        self.action_space = env.action_space

        self.config = {
            #  "batch_size": 16,
            "batch_size": 32,
            #  "batch_size": 8,
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

        #  tf.reset_default_graph()

        # self.mainQN = Qnetwork(self.config["h_size"], env.action_space.n, 'prediction')
        # self.targetQN = Qnetwork(self.config["h_size"], env.action_space.n, 'target')

        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        self.build_dqn()

        # self.w = {}
        # self.t_w = {}

        # with tf.variable_scope('pred_to_target'):
        #     self.t_w_input = {}
        #     self.t_w_assign_op = {}

        #     for name in self.w.keys():
        #         self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        #         self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        # optimizer
        # with tf.variable_scope('optimizer'):
        #     self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        #     self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        #     self.actions_onehot = tf.one_hot(self.actions, self.action_space.n, dtype=tf.float32)

        #     # self.Q = tf.reduce_sum(tf.multiply(self.mainQN.Qout, self.actions_onehot), axis=1)
        #     self.Q = tf.reduce_sum(self.mainQN.Qout * self.actions_onehot, reduction_indices=1, name='Q')

        #     self.td_error = tf.square(self.targetQ - self.Q)
        #     self.loss = tf.reduce_mean(self.td_error)
        #     # self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        #     # self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
        #     # self.trainer = tf.train.AdamOptimizer(learning_rate=0.00025)
        #     self.trainer = tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)
        #     self.optimizer = self.trainer.minimize(self.loss)

        # steps
        # with tf.variable_scope('step'):
        #     self.step_op = tf.Variable(0, trainable=False, name='step')
        #     self.step_input = tf.placeholder('int32', None, name='step_input')
        #     self.step_assign_op = self.step_op.assign(self.step_input)

        # init = tf.global_variables_initializer()

        # self.saver = tf.train.Saver()
        # self.trainables = tf.trainable_variables()
        # self.targetOps = self.updateTargetGraph(self.trainables, self.config["tau"])

        self.sess = sess
        # self.sess.run(init)
        # if self.config["load_model"]:
        #     print('Loading Model...')
        #     ckpt = tf.train.get_checkpoint_state(self.config["path"])
        #     self.saver.restore(sess, ckpt.model_checkpoint_path) # pylint: disable=E1101

        self.e = self.config["startE"]
        self.stepDrop = (self.config["startE"] - self.config["endE"]) \
            / self.config["annealing_steps"]

        self.jList = []
        self.rList = []
        self.update_count = 0
        self.total_loss = 0.
        self.total_q = 0.
        # self.loss = .0

        if not os.path.exists(self.config["path"]):
            os.makedirs(self.config["path"])

        log_path = "%s/%s/%s/%s" % (FLAGS.log_dir,
                                    FLAGS.env_name,
                                    str(self.__class__.__name__),
                                    FLAGS.timestamp)
        self.writer = tf.summary.FileWriter("%s/%s" % (log_path, '/train'), sess.graph)

    def build_dqn(self):
        # MainQ
        self.w = {}
        # self.q = {}
        self.t_w = {}
        with tf.variable_scope('main'):
            self.s_t = tf.placeholder('float32', [None, 84, 84, 4], name="s_t")
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t,
                32, [8, 8], [4, 4], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='l1')
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1,
                64, [4, 4], [2, 2], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='l2')
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2,
                64, [3, 3], [1, 1], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='l3')
            shape = self.l3.get_shape().as_list()
            self.l3_flat = tf.reshape(self.l3, [-1, functools.reduce(lambda x, y: x * y, shape[1:])])
            self.l4, self.w['l4_w'], self.w['l4_b'] = linear(self.l3_flat, 512, activation_fn=tf.nn.relu, name='l4')
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.l4, self.env.action_space.n, name='q')
            self.q_action = tf.argmax(self.q, dimension=1)

            q_summary = []
            avg_q = tf.reduce_mean(self.q, 0)
            for idx in range(self.env.action_space.n):
                q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
            self.q_summary = tf.summary.merge(q_summary, 'q_summary')

        # TargetQ
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, 84, 84, 4], name="target_s_t")
            self.t_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d(self.target_s_t,
                32, [8, 8], [4, 4], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='target_l1')
            self.t_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d(self.t_l1,
                64, [4, 4], [2, 2], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='target_l2')
            self.t_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d(self.t_l2,
                64, [3, 3], [1, 1], tf.truncated_normal_initializer(0, 0.02), tf.nn.relu, 'NHWC', name='target_l3')
            shape = self.t_l3.get_shape().as_list()
            self.t_l3_flat = tf.reshape(self.t_l3, [-1, functools.reduce(lambda x, y: x * y, shape[1:])])
            self.t_l4, self.t_w['l4_w'], self.t_w['l4_b'] = linear(self.t_l3_flat, 512, activation_fn=tf.nn.relu, name='target_l4')
            self.target_q, self.t_w['q_w'], self.t_w['q_b'] = linear(self.t_l4, self.env.action_space.n, name='target_q')
            self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
            # self.q_action = tf.argmax(self.q, 1)

        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}

            for name in self.w.keys():
                self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder(shape=[None], dtype=tf.float32, name="target_q_t")
            self.action = tf.placeholder(shape=[None], dtype=tf.int64, name= "action")
            action_onehot = tf.one_hot(self.action, self.env.action_space.n, 1.0, 0.0, name='action_onehot')
            # self.Q = tf.reduce_sum(tf.multiply(self.mainQN.Qout, self.actions_onehot), axis=1)
            q_acted = tf.reduce_sum(self.q * action_onehot, reduction_indices=1, name='Q')
            self.delta = self.target_q_t - q_acted

            # self.td_error = tf.square(self.targetQ - self.Q)
            # self.td_error = self.target_q_t - self.delta
            self.loss = tf.reduce_mean(clipped_error(self.delta), name="loss")

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
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate', 'e']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.scalar("%s" % (tag), self.summary_placeholders[tag])

            histogram_summary_tags = ['episode.rewards', 'episode.actions']

            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag]  = tf.summary.histogram(tag, self.summary_placeholders[tag])

        tf.initialize_all_variables().run()
        self.update_target_q_network()


    def learn(self, step_i, state, reward, action, done):

        # act
        # action, obs, reward, done, _ = self.act(self.env)

        self.history.add(state)
        # self.memory.add(state, action, reward, done)
        self.memory.add(state, reward, action, done)
        # print('action: ')
        # print(action)
        # print('reward: ')
        # print(reward)

        loss = .0
        if step_i > self.config["pre_train_steps"]:
            if self.memory.count < 4:
               return
            if self.e > self.config["endE"]:
                self.e -= self.stepDrop
            if step_i % (self.config["update_freq"]) == 0:
                s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()
                # print('action: ')
                # print(action)
                # print('reward: ')
                # print(reward)
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

                # pdb.set_trace()
                q_t_plus_1 = self.sess.run(self.target_q, feed_dict={
                    # self.target_input_data:np.stack(trainBatch[:, 3])
                    self.target_s_t:s_t_plus_1,
                })
                terminal = np.array(terminal) + 0.
                max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
                target_q_t = (1. - terminal) * 0.99 * max_q_t_plus_1 + reward

                _, q_t, loss = self.sess.run(
                    [self.optimizer, self.q, self.loss],
                    feed_dict={
                        self.target_q_t: target_q_t,
                        self.action: action,
                        # self.input_data:np.stack(trainBatch[:, 0]),
                        self.s_t:s_t,
                        self.learning_rate_step: step_i,
                    })
                self.total_loss += loss
                self.total_q += q_t.mean()
                self.update_count += 1


            if step_i % 500 == 499:
                self.update_target_q_network()
                # self.updateTarget(self.targetOps, self.sess)
                # self.updateTarget(self.targetOps, self.sess)

                # self.writer.add_summary(summary, self.total_steps)
                #  self.train_writer.add_summary(summary, self.total_steps)
        return self.total_loss, self.total_q, self.update_count, state, loss, self.e

    def act(self, step_i, env):
        if np.random.rand(1) < self.e or step_i < self.config["pre_train_steps"]:
            a = np.random.randint(0, self.env.action_space.n)
        else:
            # a = self.sess.run(self.q_action, feed_dict={self.input_data:np.stack(self.memory.last()[:, 3])})[0]
            # pdb.set_trace()
            a = self.sess.run(self.q_action, feed_dict={self.s_t:[self.history.get()]})[0]
        # use env rather than self.env because self.env is Gym object and env is Environemnt object
        obs, reward, done, _ = env.step(a)
        self.env.render()
        # obs = imresize(rgb2gray(obs)/255., (84, 84))
        return a, obs, reward, done, _

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars//2]):
            op_holder.append(
                tfVars[idx + total_vars//2].assign(
                    (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars//2].value())))
        return op_holder

    def updateTarget(self, op_holder, sess):
        for op in op_holder:
            sess.run(op)

    def inject_summary(self, tag_dict, step):
      summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
        self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
      })
      for summary_str in summary_str_lists:
        self.writer.add_summary(summary_str, step)

