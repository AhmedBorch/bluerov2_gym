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
env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

# Load normalization statistics if available
env = VecNormalize.load("bluerov_vec_normalize_fast.pkl", env)

# Load the pretrained model
model = PPO.load("bluerov_ppo_fast", env=env)

# Optional: continue training
model.learn(total_timesteps=200000, progress_bar=True)

# Save the updated model
model.save("bluerov_ppo_finetuned")

# Save the updated environment normalization stats
env.save("bluerov_vec_normalize_finetuned.pkl")
