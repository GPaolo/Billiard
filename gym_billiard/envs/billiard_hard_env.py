import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym_billiard.utils import physics
import Box2D as b2


import logging
logger = logging.getLogger(__name__)

class BilliardHardEnv(gym.Env):
  metadata = {'render.modes': ['human'],
              'video.frames_per_second':15
              }

  def __init__(self, seed=None):
    self.screen = None
    self.params = physics.Params()
    self.physics_eng = physics.PhysicsSim()

    # Ball XY positions can be between -1.5 and 1.5
    ball_os = spaces.Box(low=np.array([-self.params.TABLE_SIZE[0]/2., -self.params.TABLE_SIZE[1]/2.]),
                         high=np.array([self.params.TABLE_SIZE[0]/2., self.params.TABLE_SIZE[1]/2.]))

    # Arm joint can have positons:
    # Joint 0: [-Pi/2, Pi/2]
    # Joint 1: [-Pi, Pi]
    arm_joints = spaces.Box(low=np.array([-np.pi/2, -np.pi]), high=np.array([np.pi/2, np.pi]))

    self.observation_space = spaces.Dict({'ball0_position': ball_os,
                                          'ball1_position': ball_os,
                                          'arm_joints': arm_joints})

    # Actions are torques on joints and open/close of arm grip.
    # Joint torques can be between [-1, 1]
    self.action_space = spaces.Dict({'joint_torques': spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]))})

    self.seed(seed)

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def reset(self):
    init_ball0_pose = np.array([self.np_random.uniform(low=-1.5, high=1.5), # x
                                self.np_random.uniform(low=-1.5, high=0)])  # y

    init_ball1_pose = np.array([self.np_random.uniform(low=-1.5, high=1.5), # x
                                self.np_random.uniform(low=0, high=1.5)])  # y

    init_joint_pose = np.array([self.np_random.uniform(low=-np.pi/2, high=np.pi/2), # Joint0
                                self.np_random.uniform(low=-np.pi, high=np.pi)])    # Joint1

    init_joint_pose = np.zeros(2)
    self.physics_eng.reset([init_ball0_pose, init_ball1_pose], init_joint_pose)

    return self._get_obs()

  def _get_obs(self):
    '''
    This function returns the state after reading the simulator parameters.
    '''
    ball0_pose = self.physics_eng.balls[0].position + self.physics_eng.wt_transform
    ball1_pose = self.physics_eng.balls[1].position + self.physics_eng.wt_transform
    joint0 = self.physics_eng.arm['jointW0'].angle
    joint1 = self.physics_eng.arm['joint01'].angle
    self.state = {'ball0_position':np.array([ball0_pose[0], ball0_pose[1]]),
                  'ball1_position': np.array([ball1_pose[0], ball1_pose[1]]),
                  'arm_joints': np.array([joint0, joint1])}
    return self.state

  def step(self, action):
    action = np.clip(action['joint_torques'], -1, 1)

    # Set motor torques
    self.physics_eng.apply_torque_to_joint('jointW0', action[0])
    self.physics_eng.apply_torque_to_joint('joint01', action[1])
    # Simulate timestep
    self.physics_eng.step()
    #Get state
    self._get_obs()
    reward = 0

    final = False
    # Check if final state
    # Calculates if distance between the ball's center and the holes' center is smaller than the holes' radius
    ball0_pose = self.state['ball0_position']
    ball1_pose = self.state['ball1_position']

    for hole in self.physics_eng.holes:
      dist_ball0 = np.linalg.norm(ball0_pose - hole['pose'])
      dist_ball1 = np.linalg.norm(ball1_pose - hole['pose'])
      # Gets positive reward only if ball 1 goes in the hole
      # If only ball 0 goes in the hole, reward is negative
      # If both go, reward is positive but lower.
      if dist_ball1 <= hole['radius']:
        final = True
        reward += 100
      elif dist_ball0 <= hole['radius']:
        final = True
        reward += -50

    return self.state, reward, final, {}

  def render(self, mode='human', close=False):
    import pygame

    if self.screen is None:
      self.screen = pygame.display.set_mode((self.params.DISPLAY_SIZE[0], self.params.DISPLAY_SIZE[1]), 0, 32)
      pygame.display.set_caption('Billiard')
      self.clock = pygame.time.Clock()

    if self.state is None: return None

    self.screen.fill(pygame.color.THECOLORS["white"])

    # Draw holes. This are just drawn, but are not simulated.
    for hole in self.physics_eng.holes:
      # To world transform (The - is to take into account pygame coordinate system)
      pose = -hole['pose'] + self.physics_eng.tw_transform

      pygame.draw.circle(self.screen,
                         (255, 0, 0),
                         [int(pose[0] * self.params.PPM), int(pose[1] * self.params.PPM)],
                         int(hole['radius'] * self.params.PPM))

    # Draw bodies
    for body in self.physics_eng.world.bodies:
      color = [0, 0, 0]
      obj_name = body.userData['name']
      if obj_name == 'ball0':
        color = [0, 0, 180, 255]
      elif obj_name == 'ball1':
        color = [0, 180, 0, 255]
      elif obj_name in ['link0', 'link1']:
        color = [100, 100, 100]
      elif 'wall' in obj_name:
        color = [150, 150, 150]

      for fixture in body.fixtures:
        fixture.shape.draw(body, self.screen, self.params, color)

    pygame.display.flip()
    self.clock.tick(self.params.TARGET_FPS)

    return self.screen