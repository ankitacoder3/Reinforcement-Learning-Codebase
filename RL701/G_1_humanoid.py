import gym
env = gym.make("Humanoid-v4", render_mode = "human")
env.reset()

# With out training
for _ in range(1000):
    env.render()
    env.action_space
    env.observation_space
    env.step(env.action_space.sample())
env.close()

# with training
'''
healthy_reward
forward_reward
ctrl_cost
contact_cost
'''
