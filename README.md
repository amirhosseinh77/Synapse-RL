<div align="center">
  <p>
    <a align="center" href="https://github.com/amirhosseinh77/Synapse-RL">
      <img width="100%" src="https://user-images.githubusercontent.com/56114938/235786557-186fe616-f0ab-4a14-95d5-c95817062942.png"></a>
  </p>
</div>
  
# Synapse Reinforcement Learning

Synapse is a framework for implementing Reinforcement Learning (RL) algorithms in PyTorch. The repository includes popular algorithms such as Deep Q-Networks, Policy Gradients, and Actor-Critic, as well as others.

One of the advantages of using Synapse-RL is its compatibility with gym-based environments. Gym provides a standard interface for working with environments to benchmark RL models. Synapse-RL also includes various utility functions and classes that make it easy to experiment with different hyperparameters, test different training approaches, and visualize training results.

### Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/amirhosseinh77/Synapse-RL/blob/main/SYNAPES_tutorial.ipynb)

### Supported Algorithms
| RL Algorithm | Description |
| --- | --- |
| `Deep Q Learing` | Discrete |
| `Policy Gradient` | Discrete |
| `Actor Critic (A2C)` | Discrete |
| `Deep Deterministic Policy Gradient (DDGP)` | Continuous |
| `Soft Actor Critic (SAC)` | Continuous |
| `Proximal Policy Optimization (PPO)` | - |

### Inference
```python
import gym
from agents.PolicyGradient import PolicyGradientAgent

# Initialize the CartPole environment and agent
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = PolicyGradientAgent(state_size, action_size)
result = agent.train(env, episodes=1000)
```

### Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8010048.svg)](https://doi.org/10.5281/zenodo.8010048)

