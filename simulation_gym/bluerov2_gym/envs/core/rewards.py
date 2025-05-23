import numpy as np

class Reward:
    """
    Reward function for waypoint-reaching.
    get_reward(obs_dict, goal_array) returns:
    • negative Euclidean distance between current position and goal
    • plus a large bonus if within `goal_tolerance` (i.e. "reached").
    """
    
    def __init__(self, goal_tolerance: float = 0.3):
        self.goal_tolerance = goal_tolerance

    def get_reward(self, obs: dict, goal: np.ndarray) -> float:
        """
        obs: Dict containing at least keys "x","y","z" → each a (1,) array.
        goal: np.ndarray shape=(3,) holding [x_goal, y_goal, z_goal].
        Returns:
        r = −‖pos − goal‖ + (100.0 if ‖pos−goal‖ < goal_tolerance else 0).
        """
        # Extract current position
        px = float(obs["x"][0])
        py = float(obs["y"][0])
        pz = float(obs["z"][0])
        pos = np.array([px, py, pz], dtype=np.float32)
        dist = np.linalg.norm(pos - goal)
        reward = -dist
        
        # Add big bonus on "success"
        if dist < self.goal_tolerance:
            reward += 100.0
        
        return float(reward)

    def is_success(self, obs: dict, goal: np.ndarray) -> bool:
        """
        Returns True if the current position is within self.goal_tolerance of `goal`.
        """
        px = float(obs["x"][0])
        py = float(obs["y"][0])
        pz = float(obs["z"][0])
        pos = np.array([px, py, pz], dtype=np.float32)
        dist = np.linalg.norm(pos - goal)
        return dist < self.goal_tolerance
