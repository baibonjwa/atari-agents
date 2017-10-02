import sys
import logging

import numpy as np
import gym
from gym import wrappers

env = gym.make('FrozenLake-v0')

gym.undo_logger_setup()
logger = logging.getLogger()
formatter = logging.Formatter('[%(asctime)s] %(message)s')
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.addHandler(handler)

outdir = './random_agent_results/FrozenLake-v0/SimpleTabularQ/'
env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)

Q = np.zeros([env.observation_space.n, env.action_space.n])
lr = .8
y = .95

num_episodes = 1000
rList = []
for i in range(num_episodes):
  #Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    #The Q-Table learning algorithm
    while True:
        j += 1
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        #Get new state and reward from environment
        s1, r, done, _ = env.step(a)
        #Update Q-Table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        print(done)
        if done is True:
            break
    rList.append(rAll)
env.close()

print("Score over time: " +  str(sum(rList)/num_episodes))
print(Q)
