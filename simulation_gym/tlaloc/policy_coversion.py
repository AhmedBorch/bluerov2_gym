from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import bluerov2_gym  # Custom env registration

# Load and recreate environment
env = gym.make("BlueRov-v0")
env = DummyVecEnv([lambda: env])
env = VecNormalize.load("examples/bluerov_vec_normalize_good.pkl", env)
env.training = False
env.norm_reward = False

# Load model with original env
model = PPO.load("examples/bluerov_ppo_good.zip", env=env)

# Now save the model without env (safe for ROS2 loading)
model.save("tlaloc/bluerov_ppo_clean.zip")  # This won't embed the env in the model

# Save environment separately
env.save("tlaloc/bluerov_vec_normalize_clean.pkl")
