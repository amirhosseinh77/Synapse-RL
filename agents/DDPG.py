import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from utils.plot import plot_return


# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        action = F.tanh(self.fc2(x))
        return action


# Define the Q-Network architecture
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


# Define the memory buffer to store experience tuples
class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)
    
    
class DDPGAgent():
    def __init__(self, state_size, action_size, action_max, hidden_dim=128, gamma=0.99, epsilon_min=0.1, epsilon_decay=0.998, lr=1e-3, tau=0.001, buffer_size=10000, batch_size=128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.action_max = action_max
        self.epsilon = action_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)
        # actor
        self.actor = PolicyNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_actor = PolicyNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        # critic
        self.critic = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_critic = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-4)

    def select_action(self, state):
        action = self.target_actor(torch.tensor(state).to(self.device))*self.action_max
        return action + torch.randn(self.action_size).to(self.device)*self.epsilon

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.uint8).unsqueeze(-1).to(self.device)

        # Compute Q-Learning targets
        next_actions = self.target_actor(next_states)
        q_values_next = self.target_critic(next_states, next_actions)
        q_targets = rewards + (self.gamma * q_values_next * torch.logical_not(dones))
        
        # Compute Q-Learning loss and update the network parameters
        q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(q_values, q_targets.detach())
        
        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        
        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
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
                next_state, reward, done, info = env.step([action.item()])
                self.memory.push(state, action, reward, next_state, done)
                self.learn()
                score += reward
                state = next_state
            self.decay_epsilon()
            returns.append(score)
            plot_return(returns, f'Deep Deterministic Policy Gradient ({self.device})')
        env.close()
        return returns
