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
        position_error_disk = np.sqrt(
            (obs["x"][0] - obs["target_x"][0]) ** 2 + 
            (obs["y"][0] - obs["target_y"][0]) ** 2
        )
        position_error_z =  abs(obs["z"][0] - obs["target_z"][0])
        position_error = np.sqrt(obs["x"][0]**2 + obs["y"][0]**2 + obs["z"][0]**2)

        # Success bonus when close to target
        if position_error_disk < 0.7:
            buffer_zone_penalty = -3.0
        elif position_error_disk < 1.5 and position_error_disk > 0.7:
            buffer_zone_penalty = 7.0
        else:
            buffer_zone_penalty = 0.0
        
        velocity_weight = max(0.1, min(2.0, 2.0 - 0.4 * position_error_disk))
        
        # if position_error < 1:
        #     orientation_weight = 0.5
        # else:
        #     orientation_weight = 2.0

        # Velocity penalty ---> slow down when approaching the target
        velocity_penalty = np.sqrt(
            obs["vx"][0] ** 2 + obs["vy"][0] ** 2 + obs["vz"][0] ** 2
        )

        # Orientation error

        orientation_error = abs(-np.pi/2+
            np.arctan2(obs["target_x"][0] - obs["x"][0],obs["target_y"][0]- obs["y"][0] )  
        - obs["theta"][0])

        orientation_error = (orientation_error + np.pi) % (2 * np.pi) - np.pi
        orientation_error = orientation_error*180/np.pi
        orient = np.arctan2(obs["target_y"][0] - obs["y"][0],obs["target_x"][0] - obs["x"][0] )

        # Combined reward
        reward = (
            # -1.0 * position_error_disk  # Weight for position error### around 3
            # -1.0 * position_error_z # weight for z error
            - 1.0 * position_error 
            - 0.1 * velocity_penalty  # Weight for velocity
            - orientation_error  # Weight for orientation  ### pi= 3.14
            # + buffer_zone_penalty
        )
        print(f"Reward components: "
            #   f"- Position error (disk): {(-1.0 * position_error_disk):.2f}, "
            #   f"- Position error (z): {(-1.5 * position_error_z):.2f}, "
              f"- Position error: {(-1.0 * position_error):.2f}, "
              f"- Velocity penalty: {(-0.1 * velocity_penalty):.2f}, "
              f"- Orientation error: {(-orientation_error):.2f}, "
            #   f"- Buffer zone penalty: {buffer_zone_penalty:.2f}"
              f"Position: {obs['x'][0]:.2f}, {obs['y'][0]:.2f}, {obs['z'][0]:.2f}, ")
        
        print(f"x: {obs['x'][0]:.2f}, y: {obs['y'][0]:.2f}, theta: {obs['theta'][0]*180/np.pi:.2f}, orientation: {orient*180/np.pi:.2f}")

        return reward
