from importlib import resources

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer


class BlueRov(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        with resources.path("bluerov2_gym.assets", "BlueRov2.dae") as asset_path:
            self.model_path = str(asset_path)

        self.renderer = BlueRovRenderer()
        self.train = False
        # a target position instead of penalizing only from the origin
    

        # Define original waypoints
        key_points = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, -1],
            [-1, 1, -0.6],
            [-1, 0, 0.8],
            [-1, -1, -0.7],
            [0, -1, 1],
            [1, -1, 0],
            [1, 0, -1]
        ], dtype=np.float32)

        # Set how many points to generate between each pair (including endpoints)
        points_per_segment = 11  # Gives 10 steps (0.0 to 1.0 in 0.1 increments)

        # Generate interpolated path
        trajectory = []

        for i in range(len(key_points) - 1):
            start = key_points[i]
            end = key_points[i + 1]
            segment = np.linspace(start, end, num=points_per_segment, endpoint=True)
            
            # To avoid repeating points, skip the first point unless it's the first segment
            if i == 0:
                trajectory.extend(segment)
            else:
                trajectory.extend(segment[1:])

        # Convert to numpy array
        self.target_point_trajectory = np.array(trajectory, dtype=np.float32)


        self.target_idx = 0
        self.target_position = self.target_point_trajectory[0]
        self.reward_fn = Reward(self.target_position)
        # self.reward_fn = Reward()
        self.target_range = [-1, 1]
        self.dynamics = Dynamics()
        self.state = {
            "x": 0,
            "y": 0,
            "z": 0,
            "theta": 0,
            "vx": 0,
            "vy": 0,
            "vz": 0,
            "omega": 0,
        }

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "theta": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "omega": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "target_x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "target_y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "target_z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            }
        )
        self.dt = 0.1  # Time step
        self.render_mode = render_mode


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.state = {
            "x": 0,
            "y": 0,
            "z": 0,
            "theta": 0,
            "vx": 0,
            "vy": 0,
            "vz": 0,
            "omega": 0,
        }

        # Randomize the target position within the defined range (for x, y, z)
        if self.train==True:
            low, high = self.target_range
            self.target_position = np.random.uniform(low=low, high=high, size=(3,)).astype(np.float32)
            self.reward_fn = Reward(self.target_position)

        self.disturbance_dist = self.dynamics.reset()
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}
        obs["target_x"] = np.array([self.target_position[0]], dtype=np.float32)
        obs["target_y"] = np.array([self.target_position[1]], dtype=np.float32)
        obs["target_z"] = np.array([self.target_position[2]], dtype=np.float32)

        return obs, {}

    def step(self, action):
        self.dynamics.step(self.state, action)
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}
        obs["target_x"] = np.array([self.target_position[0]], dtype=np.float32)
        obs["target_y"] = np.array([self.target_position[1]], dtype=np.float32)
        obs["target_z"] = np.array([self.target_position[2]], dtype=np.float32)

        reward = self.reward_fn.get_reward(obs)

        terminated = False
        # Example conditions (please change these to your own conditions)
        if abs(self.state["z"]) > 3.0:
            terminated = True
        if abs(self.state["x"]) > 3.0 or abs(self.state["y"]) > 3.0:
            terminated = True
        if self.train==True:
            # Terminate if too close to target
            dx = self.state["x"] - self.target_position[0]
            dy = self.state["y"] - self.target_position[1]
            dz = self.state["z"] - self.target_position[2]
            distance_to_target = np.sqrt(dx**2 + dy**2 + dz**2)
            if distance_to_target < 0.2:  # Adjust threshold as needed
                terminated = True

        truncated = False

        if self.train==False:
            if reward>-0.5:
                self.target_idx=self.target_idx+1
                if self.target_idx>=len(self.target_point_trajectory):
                    self.target_idx=len(self.target_point_trajectory)-1
                self.target_position=self.target_point_trajectory[self.target_idx]
                self.reward_fn = Reward(self.target_position)


        return obs, reward, terminated, truncated, {}

    def render(self):
        self.renderer.render(self.model_path)

    def step_sim(self):
        self.renderer.step_sim(self.state)
        self.renderer.plot_target(self.target_position)