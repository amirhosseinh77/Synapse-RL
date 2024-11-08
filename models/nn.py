import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Deterministic Policy Network architecture
class DeterministicPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_max):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim
        self.action_max = torch.tensor(action_max).to(device)
        self.uncertainty = torch.tensor(action_max).to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action = F.tanh(self.fc2(x))
        return action
    
    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        action = self(state.to(device))*self.action_max
        return action + torch.randn(self.action_dim).to(device)*self.uncertainty


# Guassian Policy Network architecture
class GuassianPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_max):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        self.action_max = torch.tensor(action_max).to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_mean = torch.tanh(self.fc_mean(x))*self.action_max
        action_std = torch.clamp(self.fc_std(x), min=-20, max=2)
        action_std = torch.exp(action_std)
        return action_mean, action_std

    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        mean, std = self(state.to(device))
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        action = torch.clamp(action, min=-self.action_max, max=self.action_max)
        return action, log_prob


# Categorical Policy Network architecture
class CategoricalPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)
    
    def select_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        probs = self(state.to(device))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)


# Value Network architecture
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = self.fc2(x)
        return value


# Q-Network architecture
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        q_value = self.fc2(x)
        return q_value
    

# Deep Q-Network architecture (DQN)
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, epsilon):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.action_dim = action_dim
        self.epsilon = epsilon

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
    def select_action(self, state):
        if  torch.rand(1) <= self.epsilon:
            return torch.randint(self.action_dim, (1,))
        q_values = self(torch.tensor(state).to(device))
        return torch.argmax(q_values)
