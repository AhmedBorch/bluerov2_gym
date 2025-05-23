# reward.py

import numpy as np

class FlexibleReward:
    """
    A reward function that works in two modes:

    1) Two‐argument mode: get_reward(obs, goal)
         → identical to the original: r = −‖pos−goal‖ + (bonus if within tolerance)
    2) Three‐argument mode: get_reward(obs, goal, action)
         → extends (1) by subtracting a control‐magnitude penalty, adding an
            alignment bonus (velocity·direction), and a small time penalty each step.

    Usage:
      • Phase 1 (single waypoint): call
            r = rewarder.get_reward(obs, goal)
      • Phase 2 (two waypoints): call
            r = rewarder.get_reward(obs, goal, action)
    """

    def __init__(self,
                 goal_tolerance: float = 0.3,
                 bonus: float = 100.0,
                 ctrl_coeff: float = 0.005,
                 align_coeff: float = 0.1,
                 time_coeff: float = 0.01):
        # Common parameters
        self.goal_tolerance = goal_tolerance
        self.bonus = bonus

        # Three‐argument‐only parameters (ignored if action is None)
        self.ctrl_coeff = ctrl_coeff
        self.align_coeff = align_coeff
        self.time_coeff = time_coeff

    def get_reward(self,
                   obs: dict,
                   goal: np.ndarray,
                   action: np.ndarray = None) -> float:
        # 1) Extract current position
        px = float(obs["x"][0])
        py = float(obs["y"][0])
        pz = float(obs["z"][0])
        pos = np.array([px, py, pz], dtype=np.float32)
        eps = 1e-8

        # 2) Distance to goal
        vec_to_goal = goal - pos
        dist = float(np.linalg.norm(vec_to_goal))

        # 3) Base reward = −distance
        reward = -dist

        theta = float(obs["theta"][0])
        theta_penalty = 0.5 * abs(theta)  # Scale as needed
        reward -= theta_penalty

        # 4) Big bonus if within tolerance
        if dist < self.goal_tolerance:
            reward += self.bonus

        # If action is None, return two‐argument reward now
        if action is None:
            return float(reward)
        if action is not None:
            target_dir = goal[:2] - np.array([obs["x"][0], obs["y"][0]])
            target_dir /= np.linalg.norm(target_dir) + 1e-8
            current_dir = np.array([np.cos(theta), np.sin(theta)])
            alignment = np.dot(target_dir, current_dir)
            reward += 0.2 * max(alignment, 0)  # Bonus for facing target

        # ===== Three‐argument “shaped” additions: =====

        # 5) Control‐magnitude penalty
        #    (e.g. if action is a vector of thruster commands):
        reward -= self.ctrl_coeff * float(np.sum(np.square(action)))

        # 6) Alignment bonus: encourage velocity to point toward the goal
        #    Assumes obs has “vx” “vy” “vz” keys, each shape (1,)
        vx = float(obs["vx"][0])
        vy = float(obs["vy"][0])
        vz = float(obs["vz"][0])
        vel = np.array([vx, vy, vz], dtype=np.float32)

        # Compute unit vector from pos → goal (add small epsilon to avoid zero division)
        if dist > 1e-8:
            goal_dir_unit = vec_to_goal / (dist + eps)
        else:
            goal_dir_unit = np.zeros(3, dtype=np.float32)

        # Dot‐product between vel and goal_dir_unit
        alignment = float(np.dot(vel, goal_dir_unit))
        # Only positive alignment counts (if negative, no bonus)
        alignment_bonus = max(alignment, 0.0) * self.align_coeff
        reward += alignment_bonus

        # 7) Tiny per‐step time penalty
        reward -= self.time_coeff

        return float(reward)

    def is_success(self, obs: dict, goal: np.ndarray) -> bool:
        """
        True if ‖pos−goal‖ < goal_tolerance.
        Used by environments to decide “has reached goal?”
        """
        px = float(obs["x"][0])
        py = float(obs["y"][0])
        pz = float(obs["z"][0])
        pos = np.array([px, py, pz], dtype=np.float32)
        return float(np.linalg.norm(pos - goal)) < self.goal_tolerance
