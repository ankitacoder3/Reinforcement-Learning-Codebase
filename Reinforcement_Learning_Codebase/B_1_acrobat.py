import gym
env = gym.make("Acrobot-v1", render_mode = "human")
#check out the Acrobat  action space!
print(env.action_space)
observation = env.reset()

while True:

    env.render()
    #your agent goes here
    action = env.action_space.sample()

    observation, reward, done, info,a = env.step(action)


    if done:
      break;
print(f"reward final count {reward}")
env.close()
