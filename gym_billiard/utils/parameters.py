import numpy as np
# Params class
class Params(object):
  # Define simulation parameters (Might move them to a param file)
  # The world is centered at the lower left corner of the table
  TABLE_SIZE = np.array([3., 3.])
  TABLE_CENTER = np.array(TABLE_SIZE / 2)
  DISPLAY_SIZE = (600, 600)
  TO_PIXEL = np.array(DISPLAY_SIZE) / TABLE_SIZE

  LINK_0_LENGTH = 1.
  LINK_1_LENGTH = 1.
  LINK_ELASTICITY = 0.
  LINK_FRICTION = .9
  LINK_THICKNESS = 0.05

  BALL_RADIUS = .1
  BALL_ELASTICITY = .9
  BALL_FRICTION = .9

  WALL_THICKNESS = .05
  WALL_ELASTICITY = .95
  WALL_FRICTION = .9

  # Graphic params
  PPM = int(min(DISPLAY_SIZE)/max(TABLE_SIZE))
  TARGET_FPS = 20
  TIME_STEP = 1.0 / TARGET_FPS

  MAX_ENV_STEPS = 500
