from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EpisodeStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_rewards = []
        self.episode_velocities = []
        self.episode_counter = 0

    def _on_step(self) -> bool:
        # Get the reward for the current step
        reward = self.locals["rewards"][0]  # assuming single environment
        self.current_rewards.append(reward)

        # Check if episode is done
        done = self.locals["dones"][0]
        if done:
            ep_reward = sum(self.current_rewards)
            ep_length = len(self.current_rewards)

            self.episode_rewards.append(self.current_rewards.copy())
            self.episode_lengths.append(ep_length)
            self.episode_counter += 1


            # Logging (optional)
            print(f"Episode {self.episode_counter}: length={ep_length}, total_reward={ep_reward:.2f}")

            # Reset for next episode
            self.current_rewards.clear()
            # log the distance to the target
            obs = self.locals["new_obs"]

            # Extract and convert from array([[value]]) → float
            # x = obs["x"][0][0]
            # y = obs["y"][0][0]
            # z = obs["z"][0][0]
            tx = obs["target_x"][0][0]
            ty = obs["target_y"][0][0]
            tz = obs["target_z"][0][0]

            # Compute Euclidean distance
            distance = np.sqrt((tx)**2 + (ty)**2 + (tz)**2)

            # Append to your array
            self.episode_velocities.append(float(distance)/ep_length)

        return True  # returning False would stop training

    def get_stats(self):
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_velocities": self.episode_velocities,
        }