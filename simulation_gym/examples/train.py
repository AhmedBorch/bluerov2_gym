import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import bluerov2_gym  # This import will automatically register the environment

# Create and wrap the environment
env = gym.make("BlueRov-v0")
env.unwrapped.train = True
env = DummyVecEnv([lambda: env])
env = VecNormalize(env)


# Initialize the agent
model = PPO("MultiInputPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000,progress_bar=True)

# Save the trained model
model.save("bluerov_ppo_fast")

env.save("bluerov_vec_normalize_fast.pkl")