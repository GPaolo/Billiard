import Box2D as b2
import pygame
import numpy as np
from gym_billiard.utils import parameters
from pprint import pprint


# TODO implement different intial arm positions
# TODO implement checks on balls spawning positions (not in holes or on arm or overlapped'

# Extend polygon shape with drawing function
def draw_polygon(polygon, body, screen, params, color):
  vertices = [(body.transform * v) * params.PPM for v in polygon.vertices]
  vertices = [(v[0], params.DISPLAY_SIZE[1] - v[1]) for v in vertices]
  pygame.draw.polygon(screen, color, vertices)

b2.b2.polygonShape.draw = draw_polygon

# Extend circle shape with drawing function
def my_draw_circle(circle, body, screen, params, color):
  position = body.transform * circle.pos * params.PPM
  position = (position[0], params.DISPLAY_SIZE[1] - position[1])
  pygame.draw.circle(screen,
                     color,
                     [int(x) for x in position],
                     int(circle.radius * params.PPM))

b2.b2.circleShape.draw = my_draw_circle

class PhysicsSim(object):

  def __init__(self, balls_pose=[[0, 0]], arm_position=None, params=None):
    if params is None:
      self.params = parameters.Params()
    else:
      self.params = params

    pprint('Parameters: {}'.format(vars(self.params)))

    # Create physic simulator
    self.world = b2.b2World(gravity=(0, 0), doSleep=True)
    self.dt = 1./60
    self.vel_iter = 100
    self.pos_iter = 100
    self._create_table()
    self._create_balls(balls_pose)
    self._create_robotarm(arm_position)
    self._create_holes()

  def _create_table(self):
    '''
    Creates the walls of the table
    :return:
    '''
    # Create walls in world RF
    left_wall_body = self.world.CreateStaticBody(position=(0, self.params.TABLE_CENTER[1]),
                                                 userData={'name': 'left wall'},
                                                 shapes=b2.b2PolygonShape(box=(self.params.WALL_THICKNESS/2,
                                                                               self.params.TABLE_SIZE[1]/2)))

    right_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_SIZE[0], self.params.TABLE_CENTER[1]),
                                                  userData={'name': 'right wall'},
                                                  shapes=b2.b2PolygonShape(box=(self.params.WALL_THICKNESS/2,
                                                                                self.params.TABLE_SIZE[1] / 2)))

    upper_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_CENTER[0], self.params.TABLE_SIZE[1]),
                                                  userData={'name': 'upper wall'},
                                                  shapes=b2.b2PolygonShape(box=(self.params.TABLE_SIZE[0] / 2,
                                                                                self.params.WALL_THICKNESS/2)))
    bottom_wall_body = self.world.CreateStaticBody(position=(self.params.TABLE_CENTER[0], 0),
                                                   userData={'name': 'bottom wall'},
                                                   shapes=b2.b2PolygonShape(box=(self.params.TABLE_SIZE[0] / 2,
                                                                                 self.params.WALL_THICKNESS/2)))

    self.walls = [left_wall_body, upper_wall_body, right_wall_body, bottom_wall_body]

    # Create coordinate transform
    self.wt_transform = -self.params.TABLE_CENTER # world RF -> table RF
    self.tw_transform = self.params.TABLE_CENTER # table RF -> world RF

  def _create_balls(self, balls_pose):
    '''
    Creates the balls in the simulation at the given positions
    :param balls_pose: Initial pose of the ball in table RF
    :return:
    '''
    self.balls = []

    for idx, pose in enumerate(balls_pose):
      pose = pose + self.tw_transform # move balls in world RF
      ball = self.world.CreateDynamicBody(position=pose,
                                          bullet=True,
                                          allowSleep=True,
                                          userData={'name': 'ball{}'.format(idx)},
                                          linearDamping=1,
                                          angularDamping=1,
                                          fixtures=b2.b2FixtureDef(shape=b2.b2CircleShape(radius=self.params.BALL_RADIUS),
                                                                   density=.5,
                                                                   friction=self.params.BALL_FRICTION,
                                                                   restitution=self.params.BALL_ELASTICITY,))
      self.balls.append(ball)

  def _create_robotarm(self, arm_position=None):
    '''
    Creates the robotic arm.
    :param angular_position: Initial angular position
    :return:
    '''
    link0 = self.world.CreateDynamicBody(position=(self.params.TABLE_CENTER[0], self.params.LINK_0_LENGTH/2),
                                         bullet=True,
                                         allowSleep=True,
                                         userData={'name': 'link0'},
                                         fixtures=b2.b2FixtureDef(
                                           shape=b2.b2PolygonShape(box=(self.params.LINK_THICKNESS,
                                                                        self.params.LINK_0_LENGTH/2)),
                                           density=3,
                                           friction=self.params.LINK_FRICTION,
                                           restitution=self.params.LINK_ELASTICITY))

    # The -.1 in the position is so that the two links can overlap in order to create the joint
    link1 = self.world.CreateDynamicBody(position=(self.params.TABLE_CENTER[0], self.params.LINK_0_LENGTH - .1 + self.params.LINK_1_LENGTH / 2),
                                         bullet=True,
                                         allowSleep=True,
                                         userData={'name': 'link1'},
                                         fixtures=b2.b2FixtureDef(
                                           shape=b2.b2PolygonShape(box=(self.params.LINK_THICKNESS,
                                                                        self.params.LINK_1_LENGTH / 2)),
                                           density=3,
                                           friction=self.params.LINK_FRICTION,
                                           restitution=self.params.LINK_ELASTICITY))

    jointW0 = self.world.CreateRevoluteJoint(bodyA=self.walls[3],
                                             bodyB=link0,
                                             anchor=self.walls[3].worldCenter,
                                             lowerAngle=-.5 * b2.b2_pi,
                                             upperAngle=.5 * b2.b2_pi,
                                             enableLimit=True,
                                             maxMotorTorque=1000.0,
                                             motorSpeed=0.0,
                                             enableMotor=True)

    joint01 = self.world.CreateRevoluteJoint(bodyA=link0,
                                             bodyB=link1,
                                             anchor=link0.worldCenter + b2.b2Vec2((0, self.params.LINK_0_LENGTH/2)),
                                             lowerAngle=-b2.b2_pi,
                                             upperAngle=b2.b2_pi,
                                             enableLimit=False,
                                             maxMotorTorque=1000.0,
                                             motorSpeed=0.0,
                                             enableMotor=True)

    self.arm = {'link0': link0, 'link1': link1, 'joint01': joint01, 'jointW0': jointW0}

  def _create_holes(self):
    '''
    Defines the holes in table RF. This ones are not simulated, but just defined as a list of dicts.
    :return:
    '''
    self.holes = [{'pose': np.array([-self.params.TABLE_SIZE[0] / 2, self.params.TABLE_SIZE[1] / 2]), 'radius': .4},
                  {'pose': np.array([self.params.TABLE_SIZE[0] / 2, self.params.TABLE_SIZE[1] / 2]), 'radius': .4}]

  def reset(self, balls_pose, arm_position):
    # Destroy all the bodies
    for body in self.world.bodies:
      if body.type is b2.b2.dynamicBody:
        self.world.DestroyBody(body)

    # Recreate the balls and the arm
    self._create_balls(balls_pose)
    self._create_robotarm(arm_position)

  def move_joint(self, joint, value):
    speed = self.arm[joint].motorSpeed
    if self.params.TORQUE_CONTROL:
      self.arm[joint].motorSpeed = speed + value * self.dt
    else:
      self.arm[joint].motorSpeed = value

  def step(self):
    '''
    Performs a simulator step
    :return:
    '''
    self.world.Step(self.dt, self.vel_iter, self.pos_iter)
    self.world.ClearForces()

if __name__ == "__main__":
  phys = PhysicsSim(balls_pose=[[0, 0], [1, 1]])
  print(phys.walls[0])
