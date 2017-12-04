from __future__ import absolute_import

import importlib
import datetime
import pdb
import random
import logging
import timeit
import time
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from agents.utils import variable_summaries
from agents.memory import Memory

from inflection import underscore
from tensorflow.python import debug as tf_debug

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
        #  sess = tf_debug.LocalCLIDebugWrapperSession(sess)
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

        total_steps = agent.config["total_steps"]
        episode_count = agent.config["num_episodes"]
        max_episode_length = agent.config["max_epLength"]
        episode_rewards = []
        rewards = []

        e_tf = tf.placeholder(shape=[None], dtype=tf.float32)
        loss_tf = tf.placeholder(shape=[None], dtype=tf.float32) 
        r_tf = tf.placeholder(shape=[None], dtype=tf.float32)

        e_list = []
        loss_list = []

        variable_summaries(e_tf, 'e')
        variable_summaries(loss_tf, 'loss')
        variable_summaries(r_tf, 'reward')

        reward = 0
        done = False
        episode_num = 0

        memory = Memory()
        obs = env.reset()
        agent.env = env
        # env.render()

        #  for i in tqdm(range(episode_count)):
        for i in tqdm(range(total_steps)):


            action, obs, reward, done, _ = agent.act(memory)
            memory.add(obs, reward, action, done)
            s1, loss, e = agent.learn(obs, reward, action, done, memory)

            e_list.append(e)
            loss_list.append(loss)
            episode_rewards.append(reward)

            if done:
                episode_num += 1
                episode_rewards_sum = np.sum(episode_rewards)
                rewards.append(episode_rewards_sum)
                episode_rewards = []
                env.reset()

                merged = tf.summary.merge_all()
                summary = sess.run(merged, feed_dict={
                    r_tf: rewards,
                    e_tf: e_list,
                    loss_tf: loss_list,
                })
                agent.writer.add_summary(summary, episode_num)

    env.close()

    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. \
            If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)



if __name__ == '__main__':
    main()
