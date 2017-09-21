class TabularQAgent(object):
  def __init__(self, obs_space, action_space, **userconfig):
    if not isinstance(obs_space, discrete.Disrete):
      raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)').format(observation_space, self)
    if not isinstance(action_space, discrete.Disrete):
      raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)').format(observation_space, self)
