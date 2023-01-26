import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('CartPole-v1')

for episode in range(20):
    
    env.reset() 
    is_done = False

    while not is_done:
        action = env.action_space.sample()
        new_state, reward, is_done, info = env.step(action)
        env.render()

################# DRL #################

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

policy_net = DQN(n_observations, n_actions).to(device)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

def select_action(state):
    # global steps_done
    sample = random.random()
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #     math.exp(-1. * steps_done / EPS_DECAY)
    # steps_done += 1
    eps_threshold = 0.2
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    env.reset() 
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False

    while not done:
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if not terminated:
            state_action_values = policy_net(state).gather(1, action_batch)

            with torch.no_grad():
                next_state_values = policy_net(observation).max(1)[0]
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)




