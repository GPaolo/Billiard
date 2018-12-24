import gym
import gym_billiard
env = gym.make('Billiard-v0')
import numpy as np

print('Actions {}'.format(env.action_space))
print('Obs {}'.format(env.observation_space))

action = env.action_space.sample()
for i in range(10):
  obs = env.reset()

  for t in range(1000):
    env.render()
    # print('t {} - Obs {} - Action {}'.format(t, obs, action))
    obs, reward, done, info = env.step(action)
    print(obs[0])
    # print(obs[1])
    if done:
      print("Episode finished after {} timesteps".format(t + 1))
      break