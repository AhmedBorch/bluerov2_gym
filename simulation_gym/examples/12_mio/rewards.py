import numpy as np

class Reward:
    def __init__(self, target_position, target_orientation,desired_distance, distance_tolerance=0.2):
        self.target_position = np.array(target_position, dtype=np.float32)
        self.target_orientation = np.array(target_orientation, dtype=np.float32)
        self.desired_distance = desired_distance  # Ideal stop distance from the object
        self.distance_tolerance = distance_tolerance  # Acceptable tolerance for distance from the target
        self.stay_dest_count = 0
        
    def get_reward(self, obs):
        # Extract position, velocity, and orientation from the observation
        position = np.array([obs["x"][0], obs["y"][0], obs["z"][0]])
        velocity = np.array([obs["vx"][0], obs["vy"][0], obs["vz"][0]])
        theta = np.array(obs["theta"][0])  # Heading of the agent in radians
        omega = np.array(obs["omega"])
        z_error=np.abs(position[2]-self.target_position[2])
        # Compute the vector from the robot to the target, ignoring the z component
        vec_to_target = self.target_position - position
        distance = np.linalg.norm(vec_to_target)  # Calculate 3D distance
        
       
        
        
        distance_reward = -10 * (distance ** 2)
         # Penalize deviations from the target distance
        # Alignment: Reward dot product between heading and direction to target (in 2D)
        alignment = np.arctan2(np.sin(self.target_orientation - theta), np.cos(self.target_orientation - theta)+1e-6)
        alignment_error = np.abs(alignment)
        alignment_reward = -10 * (alignment_error)
        #print(alignment_reward,flush=True)
        
        # Velocity penalty: Reward for reducing velocity when close to the target
        velocity_penalty = -0.05 * np.linalg.norm(velocity)
        spin_penalty = -0.05 * np.linalg.norm(omega)

        success_bonus=0

        if alignment_error <  0.2:  # Apply a stronger penalty when close
            spin_penalty *= -3.0
            success_bonus+=3
        if distance <  self.distance_tolerance:  # Apply a stronger penalty when close
            velocity_penalty *= -3.0
            success_bonus+=3
        
        if alignment<0.08:
               success_bonus+=10
        if (distance < self.distance_tolerance/2) and (alignment_error < 0.2):
            success_bonus+=16  # Positive reward for "success" state
            
              #  self.stay_dest_count+=1
               # if self.stay_dest_count==30:
                #    success_bonus=1000
        #elif self.stay_dest_count>0:
         #   self.stay_dest_count-=1
        # Combined reward
        reward = alignment_reward + distance_reward - velocity_penalty - spin_penalty -z_error*3 + success_bonus
        return reward
