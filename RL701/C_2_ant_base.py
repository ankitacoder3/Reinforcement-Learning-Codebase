import gym
from stable_baselines3 import PPO
# Gyym
env = gym.make("Ant-v4",render_mode = "human")
#obs = env.reset()
#PPO code
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
episodes=10
#model.save("Ant_model")
#loaded_model = model.load("Ant_model")

for ep in range(episodes):
    obs=env.reset()
    done=False
    while not done:
        env.render()
        obs,reward,done,info= env.step(env.action_space.sample())
        #print(reward)
        
env.close()
'''
for _ in range(200):
    env.render()
    #action, _ = loaded_model.predict(obs)
    obs,done,reward,info=env.step(env.action_space.sample())
env.close()
'''