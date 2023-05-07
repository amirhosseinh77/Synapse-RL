import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=-1)
        return x

class PolicyGradientAgent():
    def __init__(self, state_size, action_size):
        self.memory = []
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.tensor(state)
        probs = self.policy_network(state)
        action = torch.distributions.Categorical(probs).sample()
        return action.item(), probs[action]

    def push_memory(self, prob, reward):
        new_experience = [torch.log(prob), reward]
        self.memory.append(new_experience)

    def learn(self, gamma=0.99):
        action_log_probs, rewards = zip(*self.memory)
        action_log_probs = torch.stack(action_log_probs)
        rewards = torch.tensor(rewards)

        discounts = torch.pow(gamma, torch.linspace(0, len(rewards)-1, len(rewards)))
        discounted_rewards = torch.flip(torch.cumsum(rewards, dim=0), [0])
        returns = torch.mul(discounts, discounted_rewards)

        # Calculate the loss 
        policy_loss = -(action_log_probs * returns).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.memory = []


# Initialize the CartPole environment and agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = PolicyGradientAgent(state_size, action_size)


# Run the Policy Gradient algorithm
episodes = 2000
returns = []
for episode in range(episodes):
    state = env.reset()
    score = 0
    done = False
    while not done:
        action, action_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.push_memory(action_prob, reward)
        score += reward
        state = next_state
    
    agent.learn()

    returns.append(score)
    plot_return(returns)
    # print("Episode: {}, Score: {:.2f}, Epsilon: {:.2f}".format(episode, score, agent.epsilon))

env.close()
plot_return(returns, show_result = True)