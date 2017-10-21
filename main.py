from __future__ import absolute_import

import importlib
import pdb
import random
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from inflection import underscore

import gym
from gym import wrappers

AVAILABLE_AGENT_LIST = [
    'RandomAgent',
    'TabularQAgent',
    'QTableLearningAgent',
    'DoubleDuelingDQNAgent',
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
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        'env_id',
        nargs='?',
        default="MsPacmanDeterministic-v0",
        help='Select the environment to run',
    )
    # parser.add_argument('agent', nargs='?', default="RandomAgent", help='Select the Agent to run')
    parser.add_argument(
        'agent',
        nargs='?',
        default="DoubleDuelingDQNAgent",
        help='Select the Agent to run'
    )
    args = parser.parse_args()

    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    env = gym.make(args.env_id)

    agent_module = importlib.import_module(underscore('agents.' + args.agent))
    agent_klass = getattr(agent_module, args.agent)

    outdir = './random_agent_results/' + args.env_id + '/' + str(agent_klass.__name__) + '/'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    with tf.Session() as sess:
        agent = agent_klass(env, sess)
        episode_count = 10000
        max_episode_length = 10000
        episode_rewards = []
        myBuffer = experience_buffer()
        rewards = []
        #  reward = 0
        #  done = False
        # this only for double_dueling_dqn_agent
        #  episodeBuffer = experience_buffer()
        #  s = processState(s)

        for i in range(episode_count):
            episodeBuffer = experience_buffer()
            obs = env.reset()
            state = processState(obs)
            reward = 0
            done = False
            for j in range(max_episode_length):
                action = agent.act(state, reward, done)
                obs, reward, done, _ = env.step(action)
                s1, loss, e = agent.learn(state, reward, action, done, episodeBuffer, myBuffer)
                state = s1
                episode_rewards.append(reward)
                if done:
                    episode_rewards_sum = np.sum(episode_rewards)
                    rewards.append(episode_rewards_sum)
                    print("episode: {0}, reward: {1}, loss: {2}, e: {3}".format(
                        i, np.sum(episode_rewards_sum), loss, e))
                    episode_rewards = []
                    break
            myBuffer.add(episodeBuffer.buffer)

    env.close()

    plt.plot(rewards)
    plt.ylabel('some numbers')
    plt.show()

    logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. \
            If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)


if __name__ == '__main__':
    main()
