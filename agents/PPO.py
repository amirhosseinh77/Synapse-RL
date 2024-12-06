import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import GuassianPolicyNetwork, ValueNetwork
from utils.asset import map_to_range, np_to_torch, torch_to_np
from utils.asset import compute_rewards_to_go
from utils.buffer import ReplayBuffer
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class PPOAgent():
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], gamma=0.99, lr=3e-4, buffer_size=1e5):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lr = lr
        self.memory = ReplayBuffer(int(buffer_size))
        self.clip_ratio = 0.2
        # actor (policy)
        self.new_policy = GuassianPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.old_policy = GuassianPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        self.old_policy.load_state_dict(self.new_policy.state_dict())
        # critic (state value)
        self.value_network = ValueNetwork(state_size, hidden_dim).to(device)
        # optimizers
        self.policy_optimizer = optim.Adam(self.new_policy.parameters(), lr=self.lr, weight_decay=1e-4)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.lr, weight_decay=1e-4)
        # log writer
        self.writer = TensorboardWriter(log_dir="Logs/PPO", comment="PPO")
        self.iter = 0

    def learn(self, gamma=0.99):
        states, action_log_probs, rewards, dones = zip(*self.memory.buffer)

        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        action_log_probs = torch.stack(action_log_probs).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones).to(device)

        # sample from old policy
        _, old_log_probs = self.old_policy.select_action(states)

        # Compute Value Targets
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

        # write loss values
        self.writer.log_scalar("Loss/Policy", policy_loss, self.iter)
        self.writer.log_scalar("Loss/Value", value_loss, self.iter)
        self.iter += 1

        # clear memory
        self.memory = ReplayBuffer(int(self.buffer_size))

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
                action_t, action_log_prob_t = self.new_policy.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                # map action to range
                mapped_action = map_to_range(action, self.action_range)
                # take action
                next_state, reward, done, _, info = env.step(mapped_action)
                # store in memory
                self.memory.push([state, action_log_prob_t, reward, done])
                state = next_state
                score += reward
                length += 1
            # train agent
            self.learn()
            # log episode info
            self.writer.log_scalar("Episode/Return", score, episode)
            self.writer.log_scalar("Episode/Length", length, episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Proximal Policy Optimization (PPO) ({device})')

        env.close()
        self.writer.close()
        return returns
