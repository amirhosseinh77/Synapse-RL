import torch
import torch.nn as nn
import torch.nn.functional as F

# Deterministic Policy Network architucture
class DeterministicPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        action = F.tanh(self.fc2(x))
        return action
    
    def select_action(self, state):
        action = self.target_actor(torch.tensor(state).to(self.device))*self.action_max
        return action + torch.randn(self.action_size).to(self.device)*self.epsilon


# Guassian Policy Network architucture
class GuassianPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_mean = F.tanh(self.fc_mean(x))
        action_std = torch.clamp(self.fc_std(x), min=-20, max=2)
        action_std = torch.exp(action_std)
        return action_mean, action_std

    def select_action(self, state):
        state = torch.tensor(state)
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        action = torch.tanh(action)
        return action, log_prob


# Value Network architucture
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