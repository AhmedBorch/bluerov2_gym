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
        # key_points = np.array([
        #     [1, 1, 1],
        #     [1, 1, 0],
        #     [0, 1, -1],
        #     [-1, 1, -0.6],
        #     [-1, 0, 0.8],
        #     [-1, -1, -0.7],
        #     [0, -1, 1],
        #     [1, -1, 0],
        #     [1, 0, -1]
        # ], dtype=np.float32)

        key_points =1*np.array([
            [1, 0, 0.4],
            [1, 1, -0.3],
            [0, 1, 0.4],
            [-1, 1, -0.3],
            [-1, 0, 0.2],
            [-1, -1, 0],
            [0, -1, 0],
            [1, -1, 0],
            [1, 0, 0],[1, 0, 0.4],
            [1, 1, -0.3],
            [0, 1, 0.4],
            [-1, 1, -0.3],
            [-1, 0, 0.2],
            [-1, -1, 0],
            [0, -1, 0],
            [1, -1, 0],
            [1, 0, 0],[1, 0, 0.4],
            [1, 1, -0.3],
            [0, 1, 0.4],
            [-1, 1, -0.3],
            [-1, 0, 0.2],
            [-1, -1, 0],
            [0, -1, 0],
            [1, -1, 0],
            [1, 0, 0]
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

        
        self.obj_idx = 0
        self.obj_pos = self.target_point_trajectory[0]
        self.target_position = self.obj_pos # for now
        
        
        
        self.target_range = [-1, 1]
        self.dynamics = Dynamics()
        self.state = {
            "x": 0,
            "y": 0,
            "z": 0,
            "theta":0,
            "vx": 0,
            "vy": 0,
            "vz": 0,
            "omega": 0,
        }

        self.target_orientation = np.array([1e-6])-np.pi/2#-np.pi/2 come from some systematic differences to make it work with the reference frame
        self.desired_distance = 0.5
        self.reward_fn = Reward(self.obj_pos,self.target_orientation,self.desired_distance)

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
                "theta": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
                "vx": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vy": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "vz": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "omega": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "target_x": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "target_y": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "target_z": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "target_angle": spaces.Box(-np.pi, np.pi, shape=(1,), dtype=np.float32),
            }
        )
        self.dt = 0.1  # Time step
        self.render_mode = render_mode
        self.random_points=False
        self.traj_counter=0
        self.base_goal_distance = 1.0  # default value

        
    

    def reset(self, *, seed=None, options=None,goal_distance=None):
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

        if self.train and goal_distance is not None:
            self.current_goal_distance = goal_distance
        else:
            self.current_goal_distance = self.base_goal_distance

        # Randomize the target position within the defined range (for x, y, z)
        if self.train==True or self.random_points==True:
            self.state["theta"] = np.random.uniform(-np.pi, np.pi)
            def sample_point_in_spherical_shell(inner_radius=0.8, outer_radius=1):
                r = ((np.random.uniform(inner_radius**3, outer_radius**3))**(1/3))
                theta = np.random.uniform(0, 2 * np.pi)
                # Bias toward cos(φ) = 0 → φ = π/2 (equator)
                #NOW
                #phi= np.pi/2
                u = np.clip(np.random.normal(loc=0.0, scale=0.5), -0.7, 0.7)  # controls bias
                phi = np.arccos(u)

                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi)

                return np.array([x, y, z], dtype=np.float32)
    
            self.obj_pos = sample_point_in_spherical_shell(inner_radius=0.5, outer_radius=1)
        help_vec = np.array([self.obj_pos[0],self.obj_pos[1],0])-np.array([self.state["x"],self.state["y"],0])
        self.target_position = self.obj_pos - help_vec / np.linalg.norm(help_vec)*self.desired_distance
        self.target_orientation = np.arctan2((self.obj_pos[1]-self.target_position[1]),(self.obj_pos[0]-self.target_position[0]+1e-6))-np.pi/2#-np.pi/2 come from some systematic differences to make it work with the reference frame
        
        self.disturbance_dist = self.dynamics.reset()
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}
        obs["target_x"] = np.array([self.target_position[0]], dtype=np.float32)
        obs["target_y"] = np.array([self.target_position[1]], dtype=np.float32)
        obs["target_z"] = np.array([self.target_position[2]], dtype=np.float32)
        obs["target_angle"] = np.array([self.target_orientation], dtype=np.float32)
        self.reward_fn = Reward(self.target_position,self.target_orientation,self.desired_distance)
        if self.train==True:
            self.reward_fn.stay_dest_count=0

        return obs, {}

    def step(self, action):
        self.dynamics.step(self.state, action)
        obs = {k: np.array([v], dtype=np.float32) for k, v in self.state.items()}
        

        if self.train==False or self.random_points==True:
            help_vec = np.array([self.obj_pos[0],self.obj_pos[1],0])-np.array([self.state["x"],self.state["y"],0])
            self.target_position = self.obj_pos - help_vec / np.linalg.norm(help_vec)*self.desired_distance
            self.target_orientation = np.arctan2((self.obj_pos[1]-self.target_position[1]),(self.obj_pos[0]-self.target_position[0]+1e-6))-np.pi/2#-np.pi/2 come from some systematic differences to make it work with the reference frame
            self.reward_fn = Reward(self.target_position,self.target_orientation,self.desired_distance)

        obs["target_x"] = np.array([self.target_position[0]], dtype=np.float32)
        obs["target_y"] = np.array([self.target_position[1]], dtype=np.float32)
        obs["target_z"] = np.array([self.target_position[2]], dtype=np.float32)
        obs["target_angle"] = np.array([self.target_orientation], dtype=np.float32)
        reward = self.reward_fn.get_reward(obs)
        
        if self.train==False:
            self.traj_counter+=1
            if reward>-14.3 and self.traj_counter%10==0:
                self.obj_idx=self.obj_idx+1
                if self.obj_idx>=len(self.target_point_trajectory):
                    self.obj_idx=len(self.target_point_trajectory)-1
                self.obj_pos=self.target_point_trajectory[self.obj_idx]

                
        terminated = False
        # Example conditions (please change these to your own conditions)
        if abs(self.state["z"]) > 10.0:
            terminated = True
        if abs(self.state["x"]) > 15.0 or abs(self.state["y"]) > 15.0:
            terminated = True

        
        
        

        if self.train==True:

                # Example conditions (please change these to your own conditions)
            if abs(self.state["z"]) > 3:
                terminated = True
            if abs(self.state["x"]) > 3 or abs(self.state["y"]) > 3:
                terminated = True

            #if self.reward_fn.stay_dest_count>50:
            #    truncated = True
            

        truncated = False
                

        return obs, reward, terminated, truncated, {}

    def render(self):
        self.renderer.render(self.model_path)

    def step_sim(self):
        self.renderer.step_sim(self.state)
        self.renderer.plot_target(self.obj_pos)
        self.renderer.plot_marker(self.target_position,orientation=self.target_orientation,marker_id="Target_state",size=7,color=0xFFFF00)