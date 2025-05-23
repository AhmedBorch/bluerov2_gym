# two_waypoint_env.py

import numpy as np
from bluerov2_gym.envs.bluerov_env import BlueRov
from reward import FlexibleReward

class TwoWaypointBlueRov(BlueRov):
    """
    Phase 2 environment: the agent must reach two sequential waypoints.
    We use FlexibleReward.get_reward(obs, goal, action) so that the
    extra control‐penalty, alignment‐bonus, and time‐penalty apply.
    """
    def __init__(self, render_mode=None, goal_tolerance=0.3, max_episode_steps=800):
        super().__init__(render_mode=render_mode,
                         goal_tolerance=goal_tolerance,
                         max_episode_steps=max_episode_steps)

        self.num_waypoints = 2

        # Instantiate the same FlexibleReward, but now we'll always call
        # get_reward(obs, goal, action) with action != None
        self.rewarder = FlexibleReward(
            goal_tolerance=goal_tolerance,
            bonus=100.0,
            ctrl_coeff=0.005,
            align_coeff=0.1,
            time_coeff=0.01
        )

    def reset(self, *, seed=None, options=None):
        """
        Sample two random goals up front, store in self.waypoints array.
        Set current_wp = 0 and goal = waypoints[0].
        """
        obs, info = super().reset(seed=seed, options=options)

        # Sample two random waypoints 
        self.waypoints = np.stack([
            [
                self.np_random.uniform(-6, 6),
                self.np_random.uniform(-6, 6),
                self.np_random.uniform(-6, 0)
            ]
            for _ in range(self.num_waypoints)
        ], axis=0).astype(np.float32)

        self.current_wp = 0
        self.goal = self.waypoints[self.current_wp]

        # If rendering, update MeshCat
        if self.render_mode == "human":
            self.renderer.reset()
            self.renderer.update_waypoint(self.goal)

        return self._build_obs_dict(), info

def step(self, action):
    """
    Handles two-waypoint navigation with shaped rewards.
    1. Advances physics while ignoring base reward
    2. Computes shaped reward with control penalties and alignment bonus
    3. Manages waypoint transitions and success flags
    """
    
    # Advance physics simulation
    obs, _, done, truncated, info = super().step(action)
    
    # Calculate shaped reward (distance + control penalty + alignment bonus)
    reward = self.rewarder.get_reward(obs, self.goal, action)

    # Waypoint transition logic
    if info.get("is_success", False):
        if self.current_wp == 0:
            # First waypoint reached: add bonus, switch to second waypoint
            reward += 50.0
            self.current_wp = 1
            self.goal = self.waypoints[self.current_wp]
            
            # Reset rewarder's internal state for new waypoint
            if hasattr(self.rewarder, 'prev_dist'):
                self.rewarder.prev_dist = None
                
            # Update visualization
            if self.render_mode == "human":
                self.renderer.update_waypoint(self.goal)
                
            # Continue episode
            info["is_success"] = False
            done = False
            
        else:
            # Second waypoint reached: let episode terminate naturally
            pass

    return obs, reward, done, truncated, info
