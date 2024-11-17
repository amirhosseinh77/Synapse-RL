import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import DQNetwork
from utils.asset import np_to_torch, torch_to_np
from utils.buffer import ReplayBuffer
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Deep Q-Learning agent
class DQNAgent():
    def __init__(self, state_size, action_size, hidden_dim=[128], gamma=0.99, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, lr=3e-4, tau=0.005, buffer_size=1e5, batch_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(int(buffer_size))
        # Q Network (state-action value)
        self.q_network = DQNetwork(state_size, action_size, hidden_dim, epsilon).to(device)
        self.target_network = DQNetwork(state_size, action_size, hidden_dim, epsilon).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr, weight_decay=1e-4)
        # log writer
        self.writer = TensorboardWriter(log_dir="Logs/DQN", comment="DQN")
        self.iter = 0

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones).to(device)
        
        # Compute action Q values 
        q_values = self.q_network(states)
        action_q_values = torch.gather(q_values, 1, actions)

        # Compute Q targets
        q_values_next = self.target_network(next_states)
        max_q_values_next = torch.max(q_values_next, dim=1)[0].unsqueeze(-1)
        q_targets = rewards + (self.gamma * max_q_values_next * torch.logical_not(dones))
        
        # Compute Q-Learning loss and update the network parameters
        loss = F.mse_loss(action_q_values, q_targets)
        
        # Update DQN network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # write loss values
        self.writer.log_scalar("Loss/DQN", loss, self.iter)
        self.iter += 1

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * target_param.data + (1-self.tau) * param.data)

    def decay_epsilon(self):
        if self.q_network.epsilon > self.epsilon_min:
            self.q_network.epsilon *= self.epsilon_decay

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
                action_t = self.q_network.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                # take action
                next_state, reward, done, _, info = env.step(action.item())
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
            self.writer.log_scalar("Episode/Epsilon", self.q_network.epsilon, episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Deep Q Learning ({device})')
        env.close()
        self.writer.close()
        return returns
