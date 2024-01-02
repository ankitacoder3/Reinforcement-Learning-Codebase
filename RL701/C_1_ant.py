import gym
env = gym.make("Ant-v4",render_mode = "human")
env.reset()
for _ in range(1000):
    env.render()
    env.action_space
    env.observation_space
    env.step(env.action_space.sample())
env.close()

# Mujoco
#reward=healthy_reward+forward_reward-ctrl_cost