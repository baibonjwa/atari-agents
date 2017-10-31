from __future__ import absolute_import

import importlib
import datetime
import hashlib
import json
import pdb
import random
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from agents.utils import variable_summaries, rgb2gray

from inflection import underscore

import gym
from tqdm import tqdm
from gym import wrappers

flags = tf.app.flags
timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

# DQN
# flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
# flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
# flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')
flags.DEFINE_string('agent_name', 'DoubleDuelingDQNAgent', 'THe name of agent to use')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')
flags.DEFINE_string('log_dir', './log', 'Value of random seed')
flags.DEFINE_string('timestamp', timestamp, 'Timestamp')
FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

AVAILABLE_AGENT_LIST = [
        'RandomAgent',
        'DoubleDuelingDQNAgent',
        'A3cFfAgent',
        'A3cLstmAgent',
        ]

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

def processState(states):
    return np.reshape(states, [states.size])

def main():
    """This is main function"""

    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    env = gym.make(FLAGS.env_name)

    with tf.Session() as sess:

        agent_module = importlib.import_module(underscore('agents.' + FLAGS.agent_name))
        agent_klass = getattr(agent_module, FLAGS.agent_name)
        agent = agent_klass(env, sess, FLAGS)

        outdir = './results/%s/%s/%s/' % (
                FLAGS.env_name,
                str(agent_klass.__name__),
                timestamp
                #  hashlib.sha224(json.dumps(agent.config).encode('utf-8')).hexdigest()
                )

        env = wrappers.Monitor(env, directory=outdir, force=True)
        env.seed(0)

        agent = agent_klass(env, sess, FLAGS)
        episode_count = 10000
        max_episode_length = 10000
        episode_rewards = []
        myBuffer = experience_buffer()
        rewards = []

        r = tf.placeholder(shape=[None], dtype=tf.float32)
        variable_summaries(r, 'reward')

        #  reward = 0
        #  done = False
        # this only for double_dueling_dqn_agent
        #  episodeBuffer = experience_buffer()
        #  s = processState(s)

        for i in tqdm(range(episode_count)):
            episodeBuffer = experience_buffer()
            obs = env.reset()
            obs = rgb2gray(obs)
            state = processState(obs)
            reward = 0
            done = False
            for j in range(max_episode_length):
                action = agent.act(state, reward, done)
                obs, reward, done, _ = env.step(action)
                obs = rgb2gray(obs)
                s1, loss, e = agent.learn(state, reward, action, done, episodeBuffer, myBuffer)
                state = s1
                episode_rewards.append(reward)
                if done:
                    episode_rewards_sum = np.sum(episode_rewards)
                    rewards.append(episode_rewards_sum)
                    episode_rewards = []
                    break
            myBuffer.add(episodeBuffer.buffer)

            merged = tf.summary.merge_all()
            summary = sess.run(merged, feed_dict={r: rewards})
            agent.writer.add_summary(summary, i)

    env.close()

    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. \
            If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)



if __name__ == '__main__':
    main()
