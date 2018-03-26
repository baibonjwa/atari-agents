import numpy as np
import random
import pdb

class Memory():
    def __init__(self, memory_size=50000):
        self.memory_size = memory_size
        self.memory = []

    def add(self, screen, reward, action, done):
        self.memory.append(np.array([screen, reward, action, done]))
        if len(self.memory) > self.memory_size:
          self.memory = self.memory[-self.memory_size:]

    def getState(self, index, size=4):
        # index = random.randint(size, self.memory_size - 1)
        index = random.randint(size, len(self.memory) - 1)
        samples = self.memory[(index - size):index]
        return np.reshape(np.array([ np.reshape(x.tolist(), (84, 84)) for x in np.array(samples)[:, 0] ]), (84, 84, 4))

    def sample(self, size):
        results = []
        for i in range(size):
            # index = random.randint(4, self.memory_size - 1)
            index = random.randint(4, len(self.memory) - 1)
            sample = self.memory[index] 
            preStates = self.getState(index - 1)
            postStates = self.getState(index)
            results.append([preStates, sample[2], sample[1], postStates, sample[3]])
        results = np.array(results)
        return results

    def last(self, size=4):
        results = []
        for i in range(size):
            # index = self.memory_size - i - 1
            index = len(self.memory) - i - 1
            sample = self.memory[index]
            preStates = self.getState(index - 1)
            postStates = self.getState(index)
            results.append([preStates, sample[2], sample[1], postStates, sample[3]])
        results = np.array(results)
        return results

    def count(self):
        return len(self.memory)
