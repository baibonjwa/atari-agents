import numpy as np
from .utils import rgb2gray
from scipy.misc import imresize

class Environment:
  def __init__(self, env):
    self.env = env

  def step(self, action):
      obs, reward, done, _ = self.env.step(action)
      obs = imresize(rgb2gray(obs)/255., (84, 84))
      # self.evn.render()
      return obs, reward, done

  def reset(self):
      obs = self.env.reset()
      obs = imresize(rgb2gray(obs)/255., (84, 84))
      return obs

  def gym_env(self):
    return self.env
