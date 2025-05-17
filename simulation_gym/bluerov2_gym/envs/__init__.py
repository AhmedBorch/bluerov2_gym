from bluerov2_gym.envs.bluerov_env import BlueRov

from gymnasium.envs.registration import register

register(
    id="BlueRov-v0",
    entry_point="bluerov2_gym.envs:BlueRov",
    max_episode_steps=100,
)
