import gym
import gym_billiard
env = gym.make('billiard-hard-v0')

print('Actions {}'.format(env.action_space))
print('Obs {}'.format(env.observation_space))

action = env.action_space.sample()

for i in range(10):
  obs = env.reset()

  for t in range(1000):
    env.render()
    # print('t {} - Obs {} - Action {}'.format(t, obs, action))
    obs, reward, done, info = env.step(action)
    print(obs['ball0_position'])
    print(obs['ball1_position'])
    if done:
      print("Episode finished after {} timesteps".format(t + 1))
      break