import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils.plot import plot_return


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)
        return F.softmax(logits, dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = self.fc2(x)
        return value

class ActorCriticAgent():
    def __init__(self, state_size, action_size, lr=1e-2, hidden_dim=128):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.memory = []
        self.actor =  Actor(state_size, hidden_dim, action_size).to(self.device)
        self.critic = Critic(state_size, hidden_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-2)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-2)


    def select_action(self, state):
        state = torch.tensor(state).to(self.device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


    def push_memory(self, state, action_log_prob, reward, next_state, done):
        new_experience = [state, action_log_prob, reward, next_state, done]
        self.memory.append(new_experience)


    def learn(self, gamma=0.99):
        states, action_log_probs, rewards, next_states, dones = zip(*self.memory)

        state_tensor = torch.tensor(np.array(states)).to(self.device)
        next_state_tensor = torch.tensor(np.array(next_states)).to(self.device)
        action_log_probs = torch.stack(action_log_probs).unsqueeze(-1).to(self.device)
        rewards = torch.tensor(rewards).unsqueeze(-1).to(self.device)
        dones = torch.tensor(dones).unsqueeze(-1).to(self.device)
        
        states_val = self.critic(state_tensor)
        next_states_val = self.critic(next_state_tensor)

        advantages = rewards + gamma * next_states_val * torch.logical_not(dones) - states_val
        actor_loss = (-action_log_probs * advantages.detach()).sum()
        critic_loss = F.mse_loss(rewards + gamma * next_states_val * torch.logical_not(dones), states_val)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()  
        self.memory = []


    def train(self, env, state_size, action_size, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action, action_log_prob = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.push_memory(state, action_log_prob, reward, next_state, done)
                score += reward
                state = next_state
            self.learn()
            returns.append(score)
            plot_return(returns, f'Actor Critic ({self.device})')
        env.close()
        return returns
