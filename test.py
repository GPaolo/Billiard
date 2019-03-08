import gym
import gym_billiard
env = gym.make('BilliardHard-v0')
import numpy as np
import matplotlib.pyplot as plt
from math import cos

print('Actions {}'.format(env.action_space))
print('Obs {}'.format(env.observation_space))

action = env.action_space.sample()
for i in range(10):
  obs = env.reset()

  for t in range(10000):
    action = [1, 1]
    img = env.render(rendered=False)
    env.render(rendered=True)
    from matplotlib import pyplot as plt

    plt.imshow(img, interpolation='nearest')
    plt.draw()
    obs, reward, done, info = env.step(action)
    if done:
      break
  print(info)
