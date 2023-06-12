import torch

def compute_rewards_to_go(rewards, gamma):
    rewards_rav = rewards.ravel()
    rewards_to_go = torch.zeros_like(rewards_rav, dtype=torch.float, device=rewards.device)
    cumulative_reward = 0
    for t in range(len(rewards_rav)-1, -1, -1):
        cumulative_reward = rewards_rav[...,t] + gamma * cumulative_reward
        rewards_to_go[t] = cumulative_reward
    return rewards_to_go.reshape(rewards.shape)
