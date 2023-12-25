# Gym  with stable baselines 3 Example

import gym
from stable_baselines3 import A2C
#import random
env = gym.make("CartPole-v1",render_mode = "human")
env.reset()

# Agent Code
model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=1000)
model.save("cartpole_model")

# Load the trained model
loaded_model = A2C.load("cartpole_model")

obs = env.reset()
#self.current_state[key] = np.array([value], dtype=int)
# Run the environment for a certain number of steps
for step in range(1000):
    # Render the environment
    env.render()

    # Get the action from the agent
    action, _ = loaded_model.predict(obs)
    # Perform the action in the environment
    
    obs, reward, done,info = env.step(action)

    # Check if the episode is done
    if done:
        break
env.close()
