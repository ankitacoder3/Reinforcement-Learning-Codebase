# Gym -1 Example
import gym
import random
import time
env = gym.make("CartPole-v1",render_mode = "human")

'''
episodes = 10
for episode in range(1,episodes+1):
  state = env.reset()
  env.step(0)
  env.observation_space
  env.observation_space.high
  env.observation_space.low
  env.action_space
  done = False
  score = 0
  while not done:
    action = random.choice([0,1])
    #action =1
    #env.render()
    a,reward,done,b,c = env.step(action)
    score += reward
    env.render()

  print(f"Episode {episode},Score {score}")
'''
episodeNumber=5
timeSteps=300
#pygame and gym 
for episodeIndex in range(episodeNumber):
    initial_state=env.reset()
    print(episodeIndex)
    env.render()
    appendedObservations=[]
    for timeIndex in range(timeSteps):
        print(timeIndex)
        random_action=env.action_space.sample()
        observation, reward, terminated, truncated, info =env.step(random_action)
        appendedObservations.append(observation)
        time.sleep(0.1)
        if (terminated):
            time.sleep(1)
            break
env.close() 
