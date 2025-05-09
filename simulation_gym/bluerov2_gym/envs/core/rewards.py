import numpy as np

class Reward:
    def __init__(self, target_position, desired_distance=0.5, distance_tolerance=0.1):
        self.target_position = np.array(target_position, dtype=np.float32)
        self.desired_distance = desired_distance  # Ideal stop distance from the object
        self.distance_tolerance = distance_tolerance  # Acceptable tolerance for distance from the target
        
    def get_reward(self, obs):
        # Extract position, velocity, and orientation from the observation
        position = np.array([obs["x"][0], obs["y"][0], obs["z"][0]])
        velocity = np.array([obs["vx"][0], obs["vy"][0], obs["vz"][0]])
        theta = np.array(obs["theta"][0])  # Heading of the agent in radians
        
        # Compute the vector from the robot to the target, ignoring the z component
        vec_to_target = self.target_position - position
        vec_to_target_2d = vec_to_target[:2]  # Only consider x and y components
        distance = np.linalg.norm(vec_to_target)  # Calculate 3D distance

        # Orientation: Heading direction vector of the agent
        heading_vec = np.array([np.cos(theta), np.sin(theta), 0])  # Only yaw, 2D vector
        vec_to_target_norm = vec_to_target_2d / (np.linalg.norm(vec_to_target_2d) + 1e-6)  # Normalize 2D vector

        # Alignment: Reward dot product between heading and direction to target (in 2D)
        alignment = np.dot(heading_vec[:2], vec_to_target_norm)  # Dot product on the 2D plane
        alignment_reward = 0
        if alignment >= 0:  # Reward for facing the target
            alignment_reward = 2 * (np.exp(alignment) - 1)  # Smooth reward curve
        else:  # Penalty for facing away from the target
            alignment_reward = -5 * np.exp(-alignment)

        # Distance reward: We want the robot to stop at a certain distance
        distance_error = np.abs(distance - self.desired_distance)
        
        if distance_error < self.distance_tolerance:
            distance_reward = 10.0  # High reward for reaching the desired distance
        else:
            distance_reward = -distance_error * 5  # Penalize deviations from the target distance
        
        # Velocity penalty: Reward for reducing velocity when close to the target
        velocity_penalty = 0.1 * np.linalg.norm(velocity)
        if distance < self.desired_distance + 0.2:  # Apply a stronger penalty when close
            velocity_penalty *= 3.0
        
        # Combined reward
        reward = alignment_reward + distance_reward - velocity_penalty
        return reward
