# train.py

import bluerov2_gym  # This import will automatically register the environment

def main():
    # Your entire training code goes here
    # Everything from make_env() to model.learn() and saving


    import time


    import gymnasium as gym
    import numpy as np
    from gymnasium.envs.registration import register
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from stable_baselines3.common.callbacks import BaseCallback # for callbacks
    

    import pickle # saving the stats
    from EpisodeStatsCallback import EpisodeStatsCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv


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
            #print(f"[Env {id(self)}] Reset at local step {self.current_step} → max_episode_steps: {self.max_episode_steps}")
            return self.env.reset(**kwargs)

        def step(self, action):
            self.current_step += 1
            self.elapsed_steps += 1
            obs, reward, terminated, truncated, info = self.env.step(action)

            if self.elapsed_steps >= self.max_episode_steps:
                truncated = True

            return obs, reward, terminated, truncated, info

    class DynamicGoalWrapper(gym.Wrapper):
        def __init__(self, env, goal_schedule_fn):
            super().__init__(env)
            self.goal_schedule_fn = goal_schedule_fn
            self.current_step = 0

        def reset(self, **kwargs):
            goal_distance = self.goal_schedule_fn(self.current_step)
            return self.env.reset(goal_distance=goal_distance, **kwargs)

        def step(self, action):
            self.current_step += 1
            return self.env.step(action)


    NUM_ENVS = 12

    def episode_length_schedule(local_step):
        approx_global_step = local_step * NUM_ENVS  # Approximate global progress
        if approx_global_step < 800_000:
            return 150
        elif approx_global_step < 1_400_000:
            return 300
        elif approx_global_step < 2_000_000:
            return 400
        else:
            return 500

    def goal_distance_schedule(local_step):
        approx_global_step = local_step * NUM_ENVS
        if approx_global_step < 2_000_000:
            return 1.0
        elif approx_global_step < 3_000_000:
            return 2.0
        elif approx_global_step < 5_000_000:
            return 3.0
        elif approx_global_step < 8_000_000:
            return 4.0
        elif approx_global_step < 10_000_000:
            return 5.0
        elif approx_global_step < 13_000_000:
            return 6.0
        else:
            return 8.0

        
    def make_env():
        def _init():
            env = gym.make("BlueRov-v0")
            env = env.unwrapped  # REMOVE default TimeLimit wrapper
            env.train = True
            env = DynamicEpisodeLengthWrapper(env, schedule_fn=episode_length_schedule)
            #env = DynamicGoalWrapper(env, goal_schedule_fn=goal_distance_schedule)
            return env
        return _init
    
    
    # Create and wrap the environment
   

    env_fns = [make_env() for _ in range(NUM_ENVS)]
    env = SubprocVecEnv(env_fns)
    #env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # Load normalization statistics if available
    #env = VecNormalize.load("examples/2mio_donut/bluerov_vec_normalize_scratch2.pkl", env)
    # Initialize PPO from scratch with MLP policy
    model = PPO("MultiInputPolicy", env,gamma=0.99,learning_rate=2.5e-4, verbose=1,
    tensorboard_log="./tensorboard_logs/")

    # Load the pretrained model
    #model = PPO.load("examples/2mio_donut/bluerov_ppo_scratch2.zip", env=env)#, device="mps")

    #training + callback initialisation

    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=2000000, callback=callback, progress_bar=True) #12mio

    # After training
    stats = callback.get_stats()
    with open("examples/training_stats.pkl", "wb") as f:
        pickle.dump(stats, f)

    # Save the updated model
    model.save("examples/bluerov_ppo_scratch3")

    # Save the updated environment normalization stats
    #env.save("examples/bluerov_vec_normalize_scratch2.pkl")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")  # Optional but safe for macOS
    main()