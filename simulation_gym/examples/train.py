import time
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
# import wandb
# from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import BaseCallback # for callbacks
import bluerov2_gym  # This import will automatically register the environment

import pickle # saving the stats
from EpisodeStatsCallback import EpisodeStatsCallback

# Create and wrap the environment
env = gym.make("BlueRov-v0")
env.unwrapped.train = True
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

# Load normalization statistics if available
env = VecNormalize.load("examples/bluerov_vec_normalize_fast.pkl", env)

# Load the pretrained model
model = PPO.load("examples/bluerov_ppo_fast", env=env)

#training + callback initialisation
callback = EpisodeStatsCallback()
model.learn(total_timesteps=1000000, callback=callback, progress_bar=True)

# After training
stats = callback.get_stats()
with open("training_stats.pkl", "wb") as f:
    pickle.dump(stats, f)


# Save the updated model
model.save("examples/bluerov_ppo_new_disk_penalty_3")

# Save the updated environment normalization stats
env.save("examples/bluerov_vec_normalize_new_disk_penalty_3.pkl")
