import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import DeterministicPolicyNetwork, QNetwork
from utils.buffer import ReplayBuffer
from utils.plot import plot_return

device = "cuda" if torch.cuda.is_available() else "cpu"
    
class DDPGAgent():
    def __init__(self, state_size, action_size, action_max, hidden_dim=128, gamma=0.99, min_uncertainty=0.1, uncertainty_decay=0.998, lr=1e-3, tau=0.001, buffer_size=10000, batch_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.min_uncertainty = min_uncertainty
        self.uncertainty_decay = uncertainty_decay
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)
        # actor
        self.actor = DeterministicPolicyNetwork(state_size, action_size, hidden_dim, action_max).to(device)
        self.target_actor = DeterministicPolicyNetwork(state_size, action_size, hidden_dim, action_max).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        # critic
        self.critic = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_critic = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=1e-4)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert data to PyTorch tensors
        states = torch.tensor(np.array(states)).to(device)
        actions = torch.tensor(actions).unsqueeze(-1).to(device)
        rewards = torch.tensor(rewards).unsqueeze(-1).to(device)
        next_states = torch.tensor(np.array(next_states)).to(device)
        dones = torch.tensor(dones).unsqueeze(-1).to(device)

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
        if self.actor.uncertainty > self.min_uncertainty:
            self.actor.uncertainty *= self.uncertainty_decay

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action = self.actor.select_action(state)
                next_state, reward, done, info = env.step([action.item()])
                self.memory.push([state, action, reward, next_state, done])
                self.learn()
                score += reward
                state = next_state
            self.decay_epsilon()
            returns.append(score)
            plot_return(returns, f'Deep Deterministic Policy Gradient (DDPG) ({device})')
        env.close()
        return returns
