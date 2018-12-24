from gym.envs.registration import register

register(
    id='billiard-v0',
    entry_point='gym_billiard.envs:BilliardEnv',
    timestep_limit=1000,
)
register(
    id='billiard-hard-v0',
    entry_point='gym_billiard.envs:BilliardHardEnv',
    timestep_limit=1000,
)