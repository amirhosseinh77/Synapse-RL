import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import random
import numpy as np
from collections import deque
from utils.plot import plot_return

# Define the Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# Define the memory buffer to store experience tuples
class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push_memory(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


# Define the Deep Q-Learning agent
class DQNAgent():
    def __init__(self, state_size, action_size, hidden_dim=128, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.998, lr=1e-3, tau=0.001, buffer_size=10000, batch_size=128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)
        self.q_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-4)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        q_values = self.q_network(torch.tensor(state, dtype=torch.float32).to(self.device))
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).unsqueeze(-1).to(self.device)
        
        # Compute Q-Learning targets
        q_values_next = self.target_network(next_states)
        max_q_values_next = torch.max(q_values_next, dim=1)[0].unsqueeze(-1)
        targets = rewards + (self.gamma * max_q_values_next * torch.logical_not(dones))
        
        # Compute Q-Learning loss and update the network parameters
        q_values = self.q_network(states)
        action_q_values = torch.gather(q_values, 1, actions)
        loss = F.mse_loss(action_q_values, targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.memory.push_memory(state, action, reward, next_state, done)
                self.learn()
                score += reward
                state = next_state
            returns.append(score)
            plot_return(returns, f'Deep Q Learning ({self.device})')
        env.close()
        return returns
