import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import CategoricalPolicyNetwork
from utils.asset import compute_rewards_to_go, np_to_torch
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class PolicyGradientAgent():
    def __init__(self, state_size, action_size, hidden_dim=[128], gamma=0.99, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.memory = []
        # actor (categorical policy)
        self.policy_network = CategoricalPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        # optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr, weight_decay=1e-4)
        # log writer
        self.writer = TensorboardWriter(log_dir="Logs/PG", comment="PG")
        self.iter = 0

    def push_memory(self, action_log_prob, reward):
        new_experience = [action_log_prob, reward]
        self.memory.append(new_experience)

    def learn(self, gamma=0.99):
        action_log_probs, rewards = zip(*self.memory)
        action_log_probs = torch.stack(action_log_probs).to(device)
        rewards = torch.tensor(rewards).unsqueeze(-1).to(device)

        discounted_returns = compute_rewards_to_go(rewards, gamma)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-6)

        # Calculate the loss 
        policy_loss = -(action_log_probs * discounted_returns).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # write loss values
        self.writer.log_scalar("Loss/Policy", policy_loss, self.iter)
        self.iter += 1

        # clear memory
        self.memory = []

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
                action, action_log_prob = self.policy_network.select_action(state_t)
                # take action
                next_state, reward, done, _, info = env.step(action.item())
                # store in memory
                self.push_memory(action_log_prob, reward)
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
            plot_return(returns, f'Policy Gradient ({device})')
        env.close()
        self.writer.close()
        return returns
