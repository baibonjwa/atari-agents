import numpy as np
from .utils import rgb2gray
from scipy.misc import imresize
import pdb

class Environment:
  def __init__(self, env):
    self.env = env
    self.lives = 0
    self.done = True

  def step(self, action):
      cumulated_reward = 0
      for _ in range(4):
        obs, reward, done, _ = self.env.step(action)
        self.done = done
        cumulated_reward = cumulated_reward + reward
        if _.get('ale.lives') < self.lives:
            cumulated_reward -= reward
            done = True
        done = _.get('ale.lives') < self.lives

        if done:
            break
      obs = imresize(rgb2gray(obs)/255., (84, 84))
      return obs, cumulated_reward, done, _

  def reset(self):
      if self.done:
        obs = self.env.reset()
      obs, reward, done, _ = self.env.step(0)
      self.lives = _.get('ale.lives')
      obs = imresize(rgb2gray(obs)/255., (84, 84))
      return obs

  def gym_env(self):
    return self.env
