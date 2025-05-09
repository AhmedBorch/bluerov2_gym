
import numpy as np

class Reward:
    def __init__(self, target_position, desired_distance=0.5):
        self.target_position = np.array(target_position, dtype=np.float32)
        self.desired_distance = desired_distance  # Ideal stop distance from the object
        self.tolerance=0.1
    def get_reward(self, obs):
        # Position and velocity
        position = np.array([obs["x"][0], obs["y"][0], obs["z"][0]])
        velocity = np.array([obs["vx"][0], obs["vy"][0], obs["vz"][0]])
        theta = obs["theta"][0]  # heading in radians

        # Distance to target
        vec_to_target = self.target_position - position
        distance = np.linalg.norm(vec_to_target)

        # Penalize deviation from desired distance (e.g., 0.5m)
        distance_error = np.abs(distance - self.desired_distance)
        #Gaussian reward centered at desired distance
        if distance >= self.desired_distance:
            # Linear reward: closer = better
            distance_reward = -5*distance  # or use -(distance - desired_distance)
        else:
            # Exponential penalty for getting too close
            penalty_strength = 10.0  # controls how sharp the penalty is
            distance_reward = -np.exp(penalty_strength * (self.desired_distance - distance))

        z_error = np.abs(position[2]-self.target_position[2])
        z_reward = -10 * z_error
        #distance_reward = -5.0 * distance_error  # penalize being too close or too far

        #ALIGNMENT OPTION 1 (deactivated for now)
        # Heading direction vector of the agent
        #heading_vec = np.array([np.cos(theta), np.sin(theta), 0])
        #vec_to_target_norm = vec_to_target / (np.linalg.norm(vec_to_target) + 1e-6)

        # Alignment: reward dot product between heading and direction to target
        
        # alignment = np.dot(heading_vec[:2], vec_to_target_norm[:2])
        # if alignment >= 0:
        #     alignment_reward = alignment  # reward in [0, 1]
        # else:
        #     alignment_reward = -np.exp(-5.0 * alignment)  # penalty grows as alignment → -1
        #OPTION 2
        # # Deviating from perfect alignment (1.0)
        # alignment_error = 1.0 - alignment  # 0 = perfect, 2 = worst case

        # # Exponential penalty on misalignment
        # alignment_penalty = -np.exp(5.0 * alignment_error)
        #OPTION 3
        #orientation_reward = 10.0 * alignment  # reward in range [-2, 2]

        # Velocity penalty, especially when close to the target
        velocity_penalty = 0.1 * np.linalg.norm(velocity)
        if distance < self.desired_distance + 0.2:
            velocity_penalty *= 3.0  # stronger penalty when close

        # Success bonus if at ideal position and aligned
        success_bonus = 0.0
        if distance_error + z_error < 0.07:# and alignment > 0.98:
            success_bonus = 10.0

        reward = distance_reward +z_reward - velocity_penalty + success_bonus #+ alignment_reward 
        return reward


# class Reward:
#     def __init__(self,  target_position):
#         # Initialize with the target position (if provided)
#         if target_position is None:
#             self.target_position = np.array([0, 0, 0], dtype=np.float32)
#         else:
#             self.target_position = target_position

#     def get_reward(self, obs):
#         # distance from the target position
#         position_error = np.sqrt(
#             (obs["x"][0] - self.target_position[0]) ** 2 + 
#             (obs["y"][0] - self.target_position[1]) ** 2 + 
#             (obs["z"][0] - self.target_position[2]) ** 2
#         )

#                 # Success bonus when close to target
#         if position_error < 0.4:
#             buffer_zone_penalty = -2.0
#         else:
#             buffer_zone_penalty = 0.0
        
#         if position_error < 0.5:
#             orientation_weight = 0.5
#         else:
#             orientation_weight = 2.0

#         # Velocity penalty ---> slow down when approaching the target
#         velocity_penalty = np.sqrt(
#             obs["vx"][0] ** 2 + obs["vy"][0] ** 2 + obs["vz"][0] ** 2
#         )

#         # Orientation error
#         orientation_error = abs(
#             np.arctan2(self.target_position[1] - obs["y"][0],self.target_position[0] - obs["x"][0] + 1e-6)  
#         - obs["theta"][0])

        

#         # Combined reward
#         reward = (
#             -1.0 * position_error  # Weight for position error### around 3
#             - 0.1 * velocity_penalty  # Weight for velocity
#             - orientation_weight * orientation_error  # Weight for orientation  ### pi= 3.14
#             + buffer_zone_penalty
#         )

#         return reward
