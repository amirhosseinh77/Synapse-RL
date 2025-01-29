import numpy as np
import random
from collections import deque

# Define the memory buffer to store experience tuples
class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, return_all=False):
        # Sample a batch of experiences
        if return_all:
            sampled_experiences = self.buffer
        else:
            sampled_experiences = random.sample(self.buffer, batch_size)
        # Transpose the list of experiences, then convert each component to a NumPy array
        sampled_experiences = [np.array(x) for x in zip(*sampled_experiences)]
        # Ensure each component has at least 2 dimensions
        return [x if x.ndim >= 2 else np.expand_dims(x, axis=-1) for x in sampled_experiences]

    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()