# one_waypoint_env.py

import numpy as np
from bluerov2_gym.envs.bluerov_env import BlueRov
from reward import FlexibleReward  # import the flexible class

class OneWaypointBlueRov(BlueRov):
    """
    Phase 1 environment: the agent must reach a single random waypoint.
    We use FlexibleReward.get_reward(obs, goal)  (action=None) so it matches
    the old two‐argument interface.
    """
    def __init__(self, render_mode=None, goal_tolerance=0.3, max_episode_steps=500):
        super().__init__(render_mode=render_mode,
                         goal_tolerance=goal_tolerance,
                         max_episode_steps=max_episode_steps)

        # Instantiate the rewarder (phase 1 will only call two‐arg get_reward)
        self.rewarder = FlexibleReward(
            goal_tolerance=goal_tolerance,
            bonus=100.0,
            ctrl_coeff=0.005,   # will be ignored when action=None
            align_coeff=0.1,    # ignored in two‐arg mode
            time_coeff=0.01     # ignored in two‐arg mode
        )

    def reset(self, *, seed=None, options=None):
        """
        Sample exactly one waypoint at the start, store it in self.goal.
        """
        obs, info = super().reset(seed=seed, options=options)

        # Sample a single random goal
        self.goal = np.array([
            self.np_random.uniform(-6, 6),
            self.np_random.uniform(-6, 6),
            self.np_random.uniform(-6, 0)
        ], dtype=np.float32)

        # If rendering, update the waypoint in MeshCat
        if self.render_mode == "human":
            self.renderer.reset()
            self.renderer.update_waypoint(self.goal)

        return self._build_obs_dict(), info

    def step(self, action):
        """
        Advance one step, compute reward only from obs and goal:
            r = get_reward(obs, goal)
        When within tolerance, the base BlueRov sets info['is_success']=True
        and done=True. We do not modify that logic further.
        """
        # 1) Advance physics; ignore the base‐class reward
        obs, _, done, truncated, info = super().step(action)

        # 2) Compute two‐arg reward = −dist + (bonus if within tol)
        #    Note: we pass action=None implicitly
        r = self.rewarder.get_reward(obs, self.goal, np.zeros(4)) 

        # 3) Return obs, r, done, truncated, info
        return obs, r, done, truncated, info
