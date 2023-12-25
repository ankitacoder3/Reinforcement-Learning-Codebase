import gym
import shimmy
from stable_baselines3 import A2C

env = gym.make("Acrobot-v1", render_mode = "human")
# training time : acrobot
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000)
model.save("Acrobot_model")

# Load the trained model
loaded_model = A2C.load("Acrobot_model")
obs = env.reset()

while True:

    env.render()
    action, _ = loaded_model.predict(obs)
    # Perform the action in the environment
    #obs, reward, done, _ = env.step(action)
    obs, reward, done, info= env.step(action)
    if done:
      break;

print(f"reward final count {reward}")
env.close()

