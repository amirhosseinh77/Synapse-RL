import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import GuassianPolicyNetwork, ValueNetwork, QNetwork
from utils.buffer import ReplayBuffer
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class SACAgent():
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], alpha=0.1, gamma=0.99, lr=3e-4, tau=0.005, buffer_size=1e5, batch_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.alpha = alpha
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(int(buffer_size))
        # actor
        self.actor = GuassianPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        # critic (state value)
        self.valueNet = ValueNetwork(state_size, hidden_dim).to(device)
        self.target_valueNet = ValueNetwork(state_size, hidden_dim).to(device)
        self.target_valueNet.load_state_dict(self.valueNet.state_dict())
        # critic (state-action value)
        self.QNet1 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.QNet2 = QNetwork(state_size, action_size, hidden_dim).to(device)
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.valueNet_optimizer = optim.Adam(self.valueNet.parameters(), lr=self.lr, weight_decay=1e-4)
        self.QNet1_optimizer = optim.Adam(self.QNet1.parameters(), lr=self.lr, weight_decay=1e-4)
        self.QNet2_optimizer = optim.Adam(self.QNet2.parameters(), lr=self.lr, weight_decay=1e-4)
        # log writer
        self.writer = TensorboardWriter(log_dir="Logs/SAC", comment="SAC")
        self.iter = 0

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, action_log_probs, rewards, next_states = self.memory.sample(self.batch_size)
        
        # Convert data to PyTorch tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1).to(device)
        action_log_probs = torch.tensor(action_log_probs, dtype=torch.float32).unsqueeze(-1).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        
        # Compute Value Targets
        state_values = self.valueNet(states)
        value_targets = torch.min(self.QNet1(states, actions), self.QNet2(states, actions)) - self.alpha*action_log_probs
        value_loss = F.mse_loss(state_values, value_targets.detach())
        
        # Update Value network
        self.valueNet_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.valueNet_optimizer.step()
        
        # Compute Q-Learning targets
        state_values_next = self.target_valueNet(next_states)
        q_targets = rewards + (self.gamma * state_values_next)
        
        q1_values = self.QNet1(states, actions)
        Q1_loss = F.mse_loss(q1_values, q_targets.detach())
        self.QNet1_optimizer.zero_grad()
        Q1_loss.backward(retain_graph=True)
        self.QNet1_optimizer.step()

        q2_values = self.QNet2(states, actions)
        Q2_loss = F.mse_loss(q2_values, q_targets.detach())
        self.QNet2_optimizer.zero_grad()
        Q2_loss.backward(retain_graph=True)
        self.QNet2_optimizer.step()
        
        # Compute actor loss
        actions, action_log_probs = self.actor.select_action(states)
        actor_loss = -(self.QNet1(states, actions) - self.alpha*action_log_probs).mean()
        
        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # write loss values
        self.writer.log_scalar("Agent Loss/Actor Loss", actor_loss, self.iter)
        self.writer.log_scalar("Agent Loss/Value Loss", value_loss, self.iter)
        self.writer.log_scalar("Agent Loss/Q1 Loss", Q1_loss, self.iter)
        self.writer.log_scalar("Agent Loss/Q2 Loss", Q2_loss, self.iter)
        self.iter += 1
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, param in zip(self.target_valueNet.parameters(), self.valueNet.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        

    def map_to_range(self, action):
        min_val, max_val = self.action_range
        mapped_action = ((action + 1) / 2) * (max_val - min_val) + min_val
        return mapped_action

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            length = 0
            done = False
            state, _ = env.reset()
            while not done:
                action, action_log_prob = self.actor.select_action(state)
                mapped_action = self.map_to_range(action.cpu().detach().numpy())
                next_state, reward, done, _, info = env.step(mapped_action)
                self.memory.push([state, action, action_log_prob, reward, next_state])
                self.learn()
                score += reward
                length += 1
                state = next_state

            self.writer.log_scalar("Episode/Return", score, episode)
            self.writer.log_scalar("Episode/Length", length, episode)

            returns.append(score)
            plot_return(returns, f'Soft Actor Critic (SAC) ({device})')

        env.close()
        self.writer.close()
        return returns
    
