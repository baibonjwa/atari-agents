class RandomAgent(object):
  def __init__(self, env):
    self.action_space = env.action_space

  def act(self, observation, reward, done):
    return self.action_space.sample()

