import time


import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from stable_baselines3.common.callbacks import BaseCallback # for callbacks
import bluerov2_gym  # This import will automatically register the environment

import pickle # saving the stats
from EpisodeStatsCallback import EpisodeStatsCallback

"""WANDB below here: if needed"""
#import wandb
#from wandb.integration.sb3 import WandbCallback


#<<<<<<< Updated upstream
# import pickle # saving the stats
# from EpisodeStatsCallback import EpisodeStatsCallback

# =======
# # keep track of the training
# wandb.init(project="bluerov-ppo", config={"learning_rate": 3e-4, "gamma": 0.99, "epochs": 10})
# >>>>>>> Stashed changes
class DynamicEpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env, schedule_fn):
        super().__init__(env)
        self.schedule_fn = schedule_fn
        self.current_step = 0
        self.max_episode_steps = self.schedule_fn(self.current_step)
        self.elapsed_steps = 0

    def reset(self, **kwargs):
        self.max_episode_steps = self.schedule_fn(self.current_step)
        self.elapsed_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step += 1
        self.elapsed_steps += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.elapsed_steps >= self.max_episode_steps:
            truncated = True

        return obs, reward, terminated, truncated, info

def episode_length_schedule(timestep):
    if timestep < 150_000:
        return 50
    elif timestep < 200_000:
        return 150
    else:
        return 200


# Create and wrap the environment
env = gym.make("BlueRov-v0")#max_episode_steps=50
env.unwrapped.train = True
env = DynamicEpisodeLengthWrapper(env, schedule_fn=episode_length_schedule)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True)

# Load normalization statistics if available
#env = VecNormalize.load("examples/200000_trained_network/bluerov_vec_normalize_scratch.pkl", env)
# Initialize PPO from scratch with MLP policy
model = PPO("MultiInputPolicy", env,gamma=0.99,learning_rate=2.5e-4, verbose=1)

# Load the pretrained model
#model = PPO.load("examples/200000_trained_network/bluerov_ppo_scratch", env=env)#, device="mps")

#training + callback initialisation

callback = EpisodeStatsCallback()
model.learn(total_timesteps=100000, callback=callback, progress_bar=True)

# After training
stats = callback.get_stats()
with open("examples/training_stats.pkl", "wb") as f:
    pickle.dump(stats, f)

# Save the updated model
model.save("examples/bluerov_ppo_scratch2")

# Save the updated environment normalization stats
env.save("examples/bluerov_vec_normalize_scratch2.pkl")
