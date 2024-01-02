import gym

env = gym.make('FrozenLake-v1', render_mode='ansi')  # build environment

current_obs = env.reset()  # start new episode
env.action_space
env.observation_space
for e in env.render():
    print(e)

new_action = env.action_space.sample()  # random action

observation, reward, done, info,a = env.step(new_action)  # perform action, ValueError!

for e in env.render():
    print(e)
'''    
0 :move left
1:down
2:right
3:up

(0,0)

current_rows*nrows+currencol
3*4+3=15

reward reach goal=1
reach hole:0
reach frozen:0
'''