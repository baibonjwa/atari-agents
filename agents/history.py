import numpy as np

class History:
  def __init__(self):
    self.history = np.zeros([4, 84, 84], dtype=np.float32)

  def add(self, screen):
    self.history[:-1] = self.history[1:]
    self.history[-1] = screen

  def reset(self):
    self.history *= 0

  def get(self):
    return np.transpose(self.history, (1, 2, 0))
