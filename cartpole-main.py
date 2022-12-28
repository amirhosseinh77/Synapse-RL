import gym

env = gym.make('CartPole-v1')

for episode in range(20):
    
    env.reset() 
    is_done = False

    while not is_done:
        action = env.action_space.sample()
        new_state, reward, is_done, info = env.step(action)
        env.render()