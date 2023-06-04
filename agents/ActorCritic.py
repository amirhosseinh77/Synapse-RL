import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import CategoricalPolicyNetwork, ValueNetwork
from utils.plot import plot_return

device = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCriticAgent():
    def __init__(self, state_size, action_size, hidden_dim=128, lr=1e-3):
        self.memory = []
        self.actor =  CategoricalPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.critic = ValueNetwork(state_size, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-4)

    def push_memory(self, state, action_log_prob, reward, next_state, done):
        new_experience = [state, action_log_prob, reward, next_state, done]
        self.memory.append(new_experience)

    def learn(self, gamma=0.99):
        states, action_log_probs, rewards, next_states, dones = zip(*self.memory)

        state_tensor = torch.tensor(np.array(states)).to(device)
        next_state_tensor = torch.tensor(np.array(next_states)).to(device)
        action_log_probs = torch.stack(action_log_probs).unsqueeze(-1).to(device)
        rewards = torch.tensor(rewards).unsqueeze(-1).to(device)
        dones = torch.tensor(dones).unsqueeze(-1).to(device)
        
        states_val = self.critic(state_tensor)
        next_states_val = self.critic(next_state_tensor)
        value_targets = rewards + gamma * next_states_val * torch.logical_not(dones)
        
        # Compute Actor Loss
        advantages = value_targets - states_val
        actor_loss = -(action_log_probs * advantages.detach()).sum()
        # Update Actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Compute Value Loss
        value_loss = F.mse_loss(states_val, value_targets)
        # Update Value network
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        self.memory = []

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action, action_log_prob = self.actor.select_action(state)
                next_state, reward, done, info = env.step(action.item())
                self.push_memory(state, action_log_prob, reward, next_state, done)
                score += reward
                state = next_state
            self.learn()
            returns.append(score)
            plot_return(returns, f'Actor Critic ({device})')
        env.close()
        return returns
