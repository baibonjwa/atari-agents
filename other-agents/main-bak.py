from __future__ import absolute_import

import importlib
import argparse
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt

from inflection import underscore

import gym
from gym import wrappers

flags = tf.app.flags

# DQN
flags.DEFINE_boolean('dueling', False, 'Whether to use dueling deep q-network')
flags.DEFINE_boolean('double_q', False, 'Whether to use double q-learning')

flags.DEFINE_string('env_name', 'Breakout-v0', 'The name of gym environment to use')
flags.DEFINE_integer('action_repeat', 4, 'The number of action to be repeated')
flags.DEFINE_string('agent_name', 'DoubleDuelingDQNAgent', 'THe name of agent to use')

# Etc
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

AVAILABLE_AGENT_LIST = [
    'RandomAgent',
    'DoubleDuelingDQNAgent',
    'A3cFfAgent',
    'A3cLstmAgent',
]

def processState(states):
    return np.reshape(states, [21168])

def main():
    """This is main function"""
    parser = argparse.ArgumentParser(description=None)

    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    env = gym.make(FLAGS.env_name)

    agent_module = importlib.import_module(underscore('agents.' + FLAGS.agent_name))
    agent_klass = getattr(agent_module, FLAGS.agent_name)

    outdir = './random_agent_results/' + FLAGS.env_name + '/' + str(agent_klass.__name__) + '/'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent = agent_klass(env)
    episode_count = 10000
    episode_rewards = []
    rewards = []
    reward = 0
    done = False
    # this only for double_dueling_dqn_agent
    #  episodeBuffer = experience_buffer()
    #  s = processState(s)

    for i in range(episode_count):
        obs = env.reset()
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            if done:
                episode_rewards_sum = np.sum(episode_rewards)
                rewards.append(episode_rewards_sum)
                print("episode: {0} {1}".format(i, np.sum(episode_rewards_sum)))
                episode_rewards = []
                break
    env.close()

    plt.plot(rewards)
    plt.ylabel('some numbers')
    plt.show()

    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. \
        If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)


if __name__ == '__main__':
    main()
