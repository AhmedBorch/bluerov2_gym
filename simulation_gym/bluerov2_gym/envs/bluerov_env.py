import numpy as np
import gymnasium as gym
from gymnasium import spaces
from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.rewards import Reward
from bluerov2_gym.envs.core.visualization.renderer import BlueRovRenderer

class BlueRov(gym.Env):
    """
    Gymnasium environment for waypoint-reaching with BlueRov2.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,  # ← Accept render_mode from calling code
        goal_tolerance: float = 0.3,
        max_episode_steps: int = 500,
    ):
        super().__init__()
        self.render_mode = render_mode
        
        # Pass render_mode into the renderer so it knows whether to open MeshCat
        self.renderer = BlueRovRenderer(render_mode=self.render_mode)
        
        # Dynamics, reward, and initial state
        self.dynamics = Dynamics()
        self.reward_fn = Reward(goal_tolerance=goal_tolerance)
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "omega": 0.0,
        }

        # Action space: 4 thruster commands ∈ [−1, +1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=+1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation: dict of current state + goal
        obs_dict = {
            "x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "theta": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "omega": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
            "x_goal": spaces.Box(-6.0, +6.0, shape=(1,), dtype=np.float32),
            "y_goal": spaces.Box(-6.0, +6.0, shape=(1,), dtype=np.float32),
            "z_goal": spaces.Box(-6.0, +0.0, shape=(1,), dtype=np.float32),
        }
        self.observation_space = spaces.Dict(obs_dict)

        # Internal bookkeeping
        self.goal = None
        self.current_step = 0
        self.max_episode_steps = max_episode_steps

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1) Sample a random goal in the desired box
        xg = self.np_random.uniform(-6.0, +6.0)
        yg = self.np_random.uniform(-6.0, +6.0)
        zg = self.np_random.uniform(-6.0, 0.0)
        self.goal = np.array([xg, yg, zg], dtype=np.float32)
        
        # 2) Reset the vehicle to "home" pose, zero velocities
        self.state = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "theta": 0.0,
            "vx": 0.0,
            "vy": 0.0,
            "vz": 0.0,
            "omega": 0.0,
        }
        _ = self.dynamics.reset()
        
        # 3) Reset step counter
        self.current_step = 0
        
        # 4) Build initial obs (contains both state and goal)
        obs = self._build_obs_dict()
        return obs, {}

    def step(self, action):
        # 1) Integrate dynamics (in-place modifies self.state)
        self.dynamics.step(self.state, action)
        
        # 2) Build observation (state + goal)
        obs = self._build_obs_dict()
        
        # 3) Compute reward
        reward = self.reward_fn.get_reward(obs, self.goal)
        
        # 4) Check success
        is_success = self.reward_fn.is_success(obs, self.goal)
        done = bool(is_success)
        
        # 5) Out-of-bounds termination
        if abs(self.state["z"]) > 10.0:
            done = True
        if abs(self.state["x"]) > 15.0 or abs(self.state["y"]) > 15.0:
            done = True
        
        # 6) Step count / truncation
        self.current_step += 1
        truncated = False
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # 7) Return info with "is_success"
        info = {"is_success": is_success}
        return obs, float(reward), done, truncated, info

    def render(self):
        # Delegate to the renderer (which only does something if render_mode=="human")
        self.renderer.render(self.model_path)

    def step_sim(self):
        # Delegate to the renderer's step_sim (again, only active in "human" mode)
        self.renderer.step_sim(self.state)

    def _build_obs_dict(self):
        obs = {}
        for key in ["x", "y", "z", "theta", "vx", "vy", "vz", "omega"]:
            obs[key] = np.array([self.state[key]], dtype=np.float32)
        obs["x_goal"] = np.array([self.goal[0]], dtype=np.float32)
        obs["y_goal"] = np.array([self.goal[1]], dtype=np.float32)
        obs["z_goal"] = np.array([self.goal[2]], dtype=np.float32)
        return obs
