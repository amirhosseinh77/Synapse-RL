import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from models.nn import CategoricalPolicyNetwork, ValueNetwork
from utils.asset import np_to_torch
from utils.plot import plot_return
from utils.logger import TensorboardWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

class ActorCriticAgent():
    def __init__(self, state_size, action_size, hidden_dim=[128], gamma=0.99, lr=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.memory = []
        # actor (categorical policy)
        self.actor =  CategoricalPolicyNetwork(state_size, action_size, hidden_dim).to(device)
        # critic (state value)
        self.critic = ValueNetwork(state_size, hidden_dim).to(device)
        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-4)
        # log writer
        self.writer = TensorboardWriter(log_dir="Logs/A2C", comment="A2C")
        self.iter = 0

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

        # write loss values
        self.writer.log_scalar("Loss/Actor", actor_loss, self.iter)
        self.writer.log_scalar("Loss/Critic", value_loss, self.iter)
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
                action, action_log_prob = self.actor.select_action(state_t)
                # take action
                next_state, reward, done, _, info = env.step(action.item())
                # store in memory
                self.push_memory(state, action_log_prob, reward, next_state, done)
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
            plot_return(returns, f'Actor Critic ({device})')
        env.close()
        self.writer.close()
        return returns
