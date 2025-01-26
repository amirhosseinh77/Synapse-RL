import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import GaussianPolicyNetwork, ValueNetwork, QNetwork
from utils.asset import map_to_range, np_to_torch, torch_to_np
from utils.buffer import ReplayBuffer
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class SACAgent():
    def __init__(self, state_size, action_size, action_range, hidden_dim=[128], 
                 alpha=0.1, gamma=0.99, lr=3e-4, tau=0.005, buffer_size=1e5, batch_size=256):
        self.state_size = state_size
        self.action_size = action_size
        self.action_range = action_range
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.batch_size = batch_size
        self.memory = ReplayBuffer(int(buffer_size))

        # Trainable entropy temperature
        self.target_entropy = -action_size  # A heuristic for continuous action spaces
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        # Actor (policy)
        self.actor = GaussianPolicyNetwork(state_size, action_size, hidden_dim).to(device)

        # Critics
        self.QNet1 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_QNet1 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_QNet1.load_state_dict(self.QNet1.state_dict())

        self.QNet2 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_QNet2 = QNetwork(state_size, action_size, hidden_dim).to(device)
        self.target_QNet2.load_state_dict(self.QNet2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=1e-4)
        self.QNet1_optimizer = optim.Adam(self.QNet1.parameters(), lr=self.lr, weight_decay=1e-4)
        self.QNet2_optimizer = optim.Adam(self.QNet2.parameters(), lr=self.lr, weight_decay=1e-4)

        # Logging
        self.writer = TensorboardWriter(log_dir="Logs/SAC_Q", comment="SAC_Q")
        self.iter = 0
        self.best_avg_reward = -np.inf

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample from replay buffer
        states, actions, action_log_probs, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convert data to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        action_log_probs = torch.tensor(action_log_probs, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Compute Q targets
        with torch.no_grad():
            next_actions, next_action_log_probs = self.actor.select_action(next_states)
            alpha = self.log_alpha.exp()  # Convert log_alpha to alpha
            q_values_next = torch.min(self.target_QNet1(next_states, next_actions), 
                                      self.target_QNet2(next_states, next_actions)) - alpha * next_action_log_probs
            q_targets = rewards + (self.gamma * q_values_next * (1 - dones))

        q_targets = q_targets.detach()

        # Update Q1
        q1_values = self.QNet1(states, actions)
        Q1_loss = F.mse_loss(q1_values, q_targets)
        self.QNet1_optimizer.zero_grad()
        Q1_loss.backward()
        self.QNet1_optimizer.step()

        # Update Q2
        q2_values = self.QNet2(states, actions)
        Q2_loss = F.mse_loss(q2_values, q_targets)
        self.QNet2_optimizer.zero_grad()
        Q2_loss.backward()
        self.QNet2_optimizer.step()

        # Compute actor loss
        actions, action_log_probs = self.actor.select_action(states)
        actor_loss = -(torch.min(self.QNet1(states, actions), self.QNet2(states, actions)) - alpha * action_log_probs).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha (train entropy)
        alpha_loss = -self.log_alpha * (action_log_probs + self.target_entropy).detach().mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Log loss values
        self.writer.log_scalar("Loss/Actor", actor_loss, self.iter)
        self.writer.log_scalar("Loss/Q1", Q1_loss, self.iter)
        self.writer.log_scalar("Loss/Q2", Q2_loss, self.iter)
        self.writer.log_scalar("Loss/Alpha", alpha_loss, self.iter)
        self.writer.log_scalar("Param/Alpha", alpha, self.iter)
        self.iter += 1

        # Soft update of target networks
        for target_param, param in zip(self.target_QNet1.parameters(), self.QNet1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_QNet2.parameters(), self.QNet2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def evaluate(self, env):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Use the policy to select an action (without exploration)
            state_t = np_to_torch(state).to(device)
            action_t, _ = self.actor.select_action(state_t, deterministic=True)
            action = torch_to_np(action_t)
            mapped_action = map_to_range(action, self.action_range)
            next_state, reward, done, _, _ = env.step(mapped_action)
            episode_reward += reward
            state = next_state

        if episode_reward > self.best_avg_reward:
            self.best_avg_reward = episode_reward
            torch.save(self.actor.state_dict(), "Logs/SAC_Q_best_actor.pth")
            print(f"New best model saved with average reward: {self.best_avg_reward}")


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
                action_t, action_log_prob_t = self.actor.select_action(state_t)
                # convert to numpy
                action = torch_to_np(action_t)
                action_log_prob = torch_to_np(action_log_prob_t)
                # map action to range
                mapped_action = map_to_range(action, self.action_range)
                # take action
                next_state, reward, done, _, info = env.step(mapped_action)
                # store in memory
                self.memory.push([state, action, action_log_prob, reward, next_state,  done])
                # train agent
                self.learn()
                state = next_state
                score += reward
                length += 1
            # log episode info
            self.writer.log_scalar("Episode/Return", score, episode)
            self.writer.log_scalar("Episode/Length", length, episode)
            # store episode return
            returns.append(score)
            plot_return(returns, f'Soft Actor Critic (SAC) ({device})')
            # Evaluation
            if (episode + 1) % 20 == 0: self.evaluate(env)
        env.close()
        self.writer.close()
        return returns
