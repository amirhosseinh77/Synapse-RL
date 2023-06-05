import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import DQNetwork
from utils.buffer import ReplayBuffer
from utils.plot import plot_return

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Deep Q-Learning agent
class DQNAgent():
    def __init__(self, state_size, action_size, hidden_dim=128, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.998, lr=1e-3, tau=0.001, buffer_size=10000, batch_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(buffer_size)
        self.q_network = DQNetwork(state_size, action_size, hidden_dim, epsilon).to(device)
        self.target_network = DQNetwork(state_size, action_size, hidden_dim, epsilon).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-4)

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
        q_values_next = self.target_network(next_states)
        max_q_values_next = torch.max(q_values_next, dim=1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * max_q_values_next * torch.logical_not(dones))
        
        # Compute Q-Learning loss and update the network parameters
        q_values = self.q_network(states)
        action_q_values = torch.gather(q_values, 1, actions)
        loss = F.mse_loss(action_q_values, q_targets.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def decay_epsilon(self):
        if self.q_network.epsilon > self.epsilon_min:
            self.q_network.epsilon *= self.epsilon_decay

    def train(self, env, episodes):
        returns = []
        for episode in range(episodes):
            score = 0
            done = False
            state = env.reset()
            while not done:
                action = self.q_network.select_action(state)
                next_state, reward, done, info = env.step(action.item())
                self.memory.push([state, action, reward, next_state, done])
                self.learn()
                score += reward
                state = next_state
            self.decay_epsilon()
            returns.append(score)
            plot_return(returns, f'Deep Q Learning ({device})')
        env.close()
        return returns
