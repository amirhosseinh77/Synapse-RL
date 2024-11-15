import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import GuassianPolicyNetwork, ValueNetwork
from utils.plot import plot_return
from utils.asset import compute_rewards_to_go
from utils.buffer import ReplayBuffer

device = "cuda" if torch.cuda.is_available() else "cpu"

class PPOAgent():
    def __init__(self, state_size, action_size, action_max, hidden_dim=128, gamma=0.99, lr=1e-2):
        self.clip_ratio = 0.2
        self.gamma = gamma
        self.lr = lr
        # self.memory = []
        buffer_size=10000
        self.memory = ReplayBuffer(buffer_size)

        # actor
        self.new_policy = GuassianPolicyNetwork(state_size, action_size, hidden_dim, action_max).to(device)
        self.old_policy = GuassianPolicyNetwork(state_size, action_size, hidden_dim, action_max).to(device)
        self.old_policy.load_state_dict(self.new_policy.state_dict())
        # critic (state value)
        self.value_network = ValueNetwork(state_size, hidden_dim).to(device)
        # optimizers
        self.policy_optimizer = optim.Adam(self.new_policy.parameters(), lr=self.lr, weight_decay=1e-4)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr, weight_decay=1e-4)

    def push_memory(self, action_log_prob, reward):
        new_experience = [action_log_prob, reward]
        self.memory.append(new_experience)

    def learn(self, gamma=0.99):
        states, action_log_probs, rewards, dones = zip(*self.memory.buffer)

        # Convert data to PyTorch tensors
        states = torch.tensor(np.array(states)).to(device)
        action_log_probs = torch.stack(action_log_probs).to(device)
        rewards = torch.tensor(rewards).unsqueeze(-1).to(device)
        dones = torch.tensor(dones).unsqueeze(-1).to(device)

        _, old_log_probs = self.old_policy.select_action(states)

        discounted_returns = compute_rewards_to_go(rewards, gamma)
        # discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-6)

        state_values = self.value_network(states)
        # Compute Value Loss
        value_loss = F.mse_loss(discounted_returns, state_values)

        # Update Value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()


        # Compute Actor Loss
        # _, action_log_probs = self.new_policy.select_action(states)

        ratios = torch.exp(action_log_probs - old_log_probs.detach())
        advantages = discounted_returns-state_values
        surr1 = ratios * advantages.detach()
        surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages.detach()
        policy_loss = -(torch.min(surr1, surr2)).mean()

        # Update old policy
        self.old_policy.load_state_dict(self.new_policy.state_dict())

        # Update actor network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.memory = ReplayBuffer(10000)

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action, action_log_prob = self.new_policy.select_action(state)
                next_state, reward, done, info = env.step([action.item()])
                self.memory.push([state, action_log_prob, reward, done])
                score += reward
                state = next_state
            self.learn()
            returns.append(score)
            plot_return(returns, f'PPO ({device})')
        env.close()
        return returns
