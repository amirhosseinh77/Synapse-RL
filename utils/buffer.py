import random
from collections import deque

# Define the memory buffer to store experience tuples
class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sampled_experiences = zip(*random.sample(self.buffer, batch_size))
        return sampled_experiences

    def __len__(self):
        return len(self.buffer)