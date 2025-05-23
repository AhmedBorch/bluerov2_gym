# train.py

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# Import your new environment wrappers
from one_waypoiont_env import OneWaypointBlueRov
from two_waypoint_env import TwoWaypointBlueRov

# ─── PPO hyperparameters ───────────────────────────────────────────────────────
TB_LOG_DIR = "tensorboard_logs/BlueRovRun"
os.makedirs(TB_LOG_DIR, exist_ok=True)

ppo_kwargs = {
    "learning_rate":   3e-4,
    "n_steps":         2048,
    "batch_size":      64,
    "n_epochs":        10,
    "gamma":           0.99,
    "gae_lambda":      0.95,
    "clip_range":      0.2,
    "ent_coef":        0.0,
    "vf_coef":         0.5,
    "max_grad_norm":   0.5,
    "tensorboard_log": TB_LOG_DIR,
    "verbose":         1,
}

# ─── Success‐rate callback ─────────────────────────────────────────────────────
class SuccessCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.success_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for info, done in zip(infos, dones):
            if done:
                self.episode_count += 1
                if info.get("is_success", False):
                    self.success_count += 1

        # At end of each rollout buffer
        pos = self.model.rollout_buffer.pos
        buf_size = self.model.rollout_buffer.buffer_size
        if pos == buf_size - 1:
            rate = (self.success_count / self.episode_count 
                    if self.episode_count > 0 else 0.0)
            self.logger.record("rollout/success_rate", rate)
            self.success_count = 0
            self.episode_count = 0
        return True

# ─── Environment factories ─────────────────────────────────────────────────────
def make_phase1_env():
    def _init():
        return OneWaypointBlueRov(
            render_mode=None,
            max_episode_steps=500
        )
    return _init

def make_phase2_env():
    def _init():
        return TwoWaypointBlueRov(
            render_mode=None,
            max_episode_steps=800
        )
    return _init

# ─── Training helper ───────────────────────────────────────────────────────────
def train_phase(env_fn, total_timesteps, model_path, init_model=None):
    # 8 parallel envs, aggregated by VecMonitor
    vec_env = SubprocVecEnv([env_fn for _ in range(16)])
    vec_env = VecMonitor(vec_env)

    if init_model:
        # Warm‐start from Phase 1, continue global step count
        model = PPO.load(
            init_model,
            env=vec_env,
            reset_num_timesteps=False
        )
    else:
        model = PPO("MultiInputPolicy", vec_env, **ppo_kwargs, device="cpu")

    model.learn(
        total_timesteps=total_timesteps,
        callback=SuccessCallback(),
        tb_log_name="BlueRovRun",
        progress_bar=True
    )
    model.save(model_path)

# ─── Main: Phase 1 → Phase 2 ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Phase 1: single waypoint
    train_phase(make_phase1_env(),
                total_timesteps=1_000_000,
                model_path="ppo_bluerov_phase1.zip",
                init_model=None)

    # Phase 2: two waypoints (warm‐start from Phase 1)
    train_phase(make_phase2_env(),
                total_timesteps=1_000_000,
                model_path="ppo_bluerov_phase2.zip",
                init_model="ppo_bluerov_phase1.zip")
