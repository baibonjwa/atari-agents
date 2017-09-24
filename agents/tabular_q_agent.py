class TabularQAgent(object):
  def __init__(self, obs_space, action_space, **userconfig):
    if not isinstance(obs_space, discrete.Disrete):
      raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)').format(obs_space, self)
    if not isinstance(action_space, discrete.Disrete):
      raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)').format(obs_space, self)
    self.obs_space = obs_space
    self.action_space = action_space
    self.action_n = action_space.n
    self.config = {
        "init_mean" : 0.0,
        "init_std" : 0.0,
        "learning_rate" : 0.1,
        "eps" : '0.05',
        "discount" : 0.95,
        "n_iter": 10000,
    }
    self.config.update(userconfig)
    self.q = defaultdict(lambda: self.config['init_std'] * np.random.randn(self.action_n) + self.config["init_mean"])

  def act(self, obs, eps = None):
    if eps is None:
      eps = self.config["eps"]
    # epsilon greedy
    action = np.argmax(self.q[observation.item()]) if np.random.random() > eps else self.action_space.sample()
    return action

  def learn(self, env):
    config = self.config
    obs = env.reset()
    q = self.q
    for t in range(config["n_iter"]):
      action, _ = self.act(obs)
      obs2, reward, done, _ = env.step(action)
      future = 0.0
      if not done:
        future = np.max(q[obs2.item()])
      q[obs.item()][action] -= \
        self.config["learning_rate"] * (q[obs.item()][action] - reward - config["discount"] * future)

      obs = obs2
