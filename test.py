import gym
import gym_billiard
env = gym.make('Billiard-v0')
import numpy as np
import matplotlib.pyplot as plt
from math import cos

print('Actions {}'.format(env.action_space))
print('Obs {}'.format(env.observation_space))

action = env.action_space.sample()
for i in range(10):
  obs = env.reset()

  for t in range(10000):
    action = [0, 0.1]
    if t % 50 == 0:
      # action = env.action_space.sample()
      print('Action {}'.format(action))
    env.render()
    # print('t {} - Obs {} - Action {}'.format(t, obs, action))
    obs, reward, done, info = env.step(action)
    # print('Ball Vel {}'.format(obs[2]))
    # print(obs[1])
    # if done:
      # print("Episode finished after {} timesteps".format(t + 1))
      # break