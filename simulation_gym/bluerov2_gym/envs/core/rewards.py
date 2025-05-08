import numpy as np


class Reward:
    def __init__(self,  target_position):
        # Initialize with the target position (if provided)
        if target_position is None:
            self.target_position = np.array([0, 0, 0], dtype=np.float32)
        else:
            self.target_position = target_position

    def get_reward(self, obs):
        # distance from the target position
        position_error = np.sqrt(
            (obs["x"][0] - self.target_position[0]) ** 2 + 
            (obs["y"][0] - self.target_position[1]) ** 2 + 
            (obs["z"][0] - self.target_position[2]) ** 2
        )

        # Velocity penalty ---> slow down when approaching the target
        velocity_penalty = np.sqrt(
            obs["vx"][0] ** 2 + obs["vy"][0] ** 2 + obs["vz"][0] ** 2
        )

        # Orientation error
        orientation_error = abs(np.arctan2((self.target_position[1]-obs["y"][0]),(self.target_position[0]-obs["x"][0]))-obs["theta"][0])

        # Combined reward
        reward = -(
            1.0 * position_error  # Weight for position error
            + 0.1 * velocity_penalty  # Weight for velocity
            + 1 * orientation_error  # Weight for orientation
        )

        return reward
