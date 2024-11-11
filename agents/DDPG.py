import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import DeterministicPolicyNetwork, QNetwork
from utils.asset import map_to_range, np_to_torch, torch_to_np
from utils.buffer import ReplayBuffer
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"
    
class DDPGAgent():
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], gamma=0.99, min_uncertainty=0.1, uncertainty_decay=0.998, lr=3e-4, tau=0.005, buffer_size=1e5, batch_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.min_uncertainty = min_uncertainty
        self.uncertainty_decay = uncertainty_decay
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)
        # actor (policy)
        self.actor = DeterministicPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_actor = DeterministicPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        # critic (state-action value)
        self.critic = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_critic = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-4)
        # log writer
        self.writer = TensorboardWriter(log_dir="Logs/DDPG", comment="DDPG")
        self.iter = 0

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones).to(device)

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

        # write loss values
        self.writer.log_scalar("Loss/Actor", actor_loss, self.iter)
        self.writer.log_scalar("Loss/Critic", critic_loss, self.iter)
        self.iter += 1

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def decay_epsilon(self):
        # self.actor.uncertainty[self.actor.uncertainty > self.min_uncertainty] *= self.uncertainty_decay
        self.actor.uncertainty = torch.minimum(self.actor.uncertainty*self.uncertainty_decay, self.actor.uncertainty)

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            length = 0
            done = False
            state, _ = env.reset()
            while not done:
                # convert to tensor
                state_t = np_to_torch(state).to(device)
                # select action
                action_t = self.actor.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                # map action to range
                mapped_action = map_to_range(action, self.action_range)
                # take action
                next_state, reward, done, _, info = env.step(mapped_action)
                # store in memory
                self.memory.push([state, action, reward, next_state, done])
                # train agent
                self.learn()
                state = next_state
                score += reward
                length += 1
            # decrease exploration
            self.decay_epsilon()
            # log episode info
            self.writer.log_scalar("Episode/Return", score, episode)
            self.writer.log_scalar("Episode/Length", length, episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Deep Deterministic Policy Gradient (DDPG) ({device})')
        env.close()
        self.writer.close()
        return returns