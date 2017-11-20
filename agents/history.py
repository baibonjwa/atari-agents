import numpy as np
import random
import pdb

class History():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) -  self.buffer_size] = []
        self.buffer.extend(experience)

    def getState(self, index):
        index = random.randint(4, self.buffer_size)
        samples = self.buffer[(index - 4):index]
        return np.reshape(np.array([ np.reshape(x.tolist(), (84, 84)) for x in np.array(samples)[:, 0] ]), (84, 84, 4))

    def sample(self, size):
        results = []
        for i in range(size):
            index = random.randint(4, self.buffer_size)
            sample = self.buffer[index] 
            preStates = self.getState(index - 1)
            postStates = self.getState(index)
            results.append([preStates, sample[1], sample[2], postStates, sample[4]])
        results = np.array(results)
        return results
