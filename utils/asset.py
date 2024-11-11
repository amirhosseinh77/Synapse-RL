import torch

def compute_rewards_to_go(rewards, gamma):
    rewards_rav = rewards.ravel()
    rewards_to_go = torch.zeros_like(rewards_rav, dtype=torch.float, device=rewards.device)
    cumulative_reward = 0
    for t in range(len(rewards_rav)-1, -1, -1):
        cumulative_reward = rewards_rav[...,t] + gamma * cumulative_reward
        rewards_to_go[t] = cumulative_reward
    return rewards_to_go.reshape(rewards.shape)

def map_to_range(action, range):
    min_val, max_val = range
    mapped_action = ((action + 1) / 2) * (max_val - min_val) + min_val
    return mapped_action

def np_to_torch(x):
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

def torch_to_np(x):
    return x.squeeze(0).cpu().detach().numpy().ravel()
