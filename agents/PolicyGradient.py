import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import CategoricalPolicyNetwork
from utils.asset import compute_rewards_to_go
from utils.plot import plot_return

device = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyGradientAgent():
    def __init__(self, state_size, action_size, hidden_dim=128, lr=1e-3):
        self.memory = []
        self.policy_network = CategoricalPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr, weight_decay=1e-4)

    def push_memory(self, action_log_prob, reward):
        new_experience = [action_log_prob, reward]
        self.memory.append(new_experience)

    def learn(self, gamma=0.99):
        action_log_probs, rewards = zip(*self.memory)
        action_log_probs = torch.stack(action_log_probs).to(device)
        rewards = torch.tensor(rewards).to(device)

        discounted_returns = compute_rewards_to_go(rewards, gamma)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-6)

        # Calculate the loss 
        policy_loss = -(action_log_probs * discounted_returns).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()

        self.optimizer.step()
        self.memory = []

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action, action_log_prob = self.policy_network.select_action(state)
                next_state, reward, done, info = env.step(action.item())
                self.push_memory(action_log_prob, reward)
                score += reward
                state = next_state
            self.learn()
            returns.append(score)
            plot_return(returns, f'Policy Gradient ({device})')
        env.close()
        return returns
