import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils.plot import plot_return


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)


class PolicyGradientAgent():
    def __init__(self, state_size, action_size, lr=1e-3, hidden_dim=128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory = []
        self.policy_network = PolicyNetwork(state_size, hidden_dim, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, weight_decay=1e-4)
        self.max_gradient_norm = 0.5

    def select_action(self, state):
        state = torch.tensor(state).to(self.device)
        probs = self.policy_network(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def push_memory(self, action_log_prob, reward):
        new_experience = [action_log_prob, reward]
        self.memory.append(new_experience)

    def learn(self, gamma=0.99):
        action_log_probs, rewards = zip(*self.memory)
        action_log_probs = torch.stack(action_log_probs).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)

        discounts = torch.pow(gamma, torch.linspace(0, len(rewards)-1, len(rewards))).to(self.device)
        discounted_rewards = torch.flip(torch.cumsum(rewards, dim=0), [0])
        returns = torch.mul(discounts, discounted_rewards)

        # Calculate the loss 
        policy_loss = -(action_log_probs * returns).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()

        # gradients clipping
        # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.max_gradient_norm)

        self.optimizer.step()
        self.memory = []

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action, action_log_prob = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.push_memory(action_log_prob, reward)
                score += reward
                state = next_state
            self.learn()
            returns.append(score)
            plot_return(returns, f'Policy Gradient ({self.device})')
        env.close()
        return returns
