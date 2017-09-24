from __future__ import absolute_import

import importlib
from inflection import underscore
import argparse
import logging
import sys
import gym
import pdb
import numpy as np
import matplotlib.pyplot as plt

from gym import wrappers

AVAILABLE_AGENT_LIST = [
  'RandomAgent',
  'TabularQAgent',
  'QTableLearningAgent',
]

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = None)
  parser.add_argument('env_id', nargs='?', default="MsPacmanDeterministic-v0", help='Select the environment to run')
  parser.add_argument('agent', nargs='?', default="RandomAgent", help='Select the Agent to run')
  args = parser.parse_args()

  gym.undo_logger_setup()
  logger = logging.getLogger()
  formatter = logging.Formatter('[%(asctime)s] %(message)s')
  handler = logging.StreamHandler(sys.stderr)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  logger.setLevel(logging.INFO)

  env = gym.make(args.env_id)

  outdir = './random_agent_results'
  env = wrappers.Monitor(env, directory = outdir, force = True)
  env.seed(0)

  agent_module = importlib.import_module(underscore('agents.' + args.agent))
  agent_klass = getattr(agent_module, args.agent)
  agent = agent_klass(env)
  episode_count = 100
  episode_rewards = []
  rewards = []
  reward = 0
  done = False

  for i in range(episode_count):
    ob = env.reset()
    while True:
      action = agent.act(ob, reward, done)
      ob, reward, done, _ = env.step(action)
      episode_rewards.append(reward)
      if done:
        episode_rewards_sum = np.sum(episode_rewards)
        rewards.append(episode_rewards_sum)
        print(np.sum(episode_rewards_sum))
        episode_rewards = []
        break
  env.close()

  plt.plot(rewards)
  plt.ylabel('some numbers')
  plt.show()

  logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
  # gym.upload(outdir)
