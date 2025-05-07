import time

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register, registry
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import bluerov2_gym  # This import will automatically register the environment


import webbrowser

import threading
import requests
import cv2

import subprocess

import os

def start_camera_server():
    script_path = os.path.join(os.path.dirname(__file__), "camera_server.py")
    print(f"🚀 Starting camera server from: {script_path}")
    
    process = subprocess.Popen(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True  # Decode bytes to strings
    )

    def log_output(stream, name):
        for line in iter(stream.readline, ''):
            print(f"[{name}] {line.strip()}")
        stream.close()

    # Start threads to log stdout and stderr
    threading.Thread(target=log_output, args=(process.stdout, "STDOUT"), daemon=True).start()
    threading.Thread(target=log_output, args=(process.stderr, "STDERR"), daemon=True).start()
    
    return process

def open_camera_tab():
    webbrowser.open("http://127.0.0.1:5050")
    
def stop_camera_server(process):
    """Stop the camera server by terminating the subprocess."""
    print("🛑 Stopping camera server...")
    process.terminate()  # Gracefully terminate the camera server
    process.wait()  # Wait for the process to fully terminate

def test_agent():

    # Start the camera server
    camera_server_process = start_camera_server()

    # Create the environment with rendering enabled
    env = gym.make("BlueRov-v0", render_mode="human",max_episode_steps=400)

    # Load the trained model and normalization stats
    model = PPO.load("examples/200000_trained_network/bluerov_ppo_fast")

    # Create a dummy vec env for proper normalization
    vec_env = DummyVecEnv([lambda: gym.make("BlueRov-v0")])
    vec_env = VecNormalize.load("examples/200000_trained_network/bluerov_vec_normalize_fast.pkl", vec_env) #forgot to save it, skip for now

    # Configure normalization for inference
    vec_env.training = False
    vec_env.norm_reward = False

    # Run episodes
    episodes = 1  # Number of episodes to visualize

    test_i=0


    for episode in range(episodes):
        obs, _ = env.reset()
        env.render()  # Initial render
        
        episode_reward = 0
        step_count = 0

        print(f"\nStarting Episode {episode + 1}")

        while True:

            # pblishing positional data to the html, so we can get it when rnning the camera server
            position = np.array([env.unwrapped.state["x"],env.unwrapped.state["y"],env.unwrapped.state["z"]])
            orientation = np.array([0,env.unwrapped.state["theta"],env.unwrapped.state["omega"]])
            target_buoy =np.array([env.unwrapped.target_position])
            trajectory_markers = np.array([env.unwrapped.renderer.trail_positions])# we want the position of the traectory points
            
            try:
                requests.post("http://127.0.0.1:5050/update_state", json={
                    "position": position.tolist(),
                    "orientation": orientation.tolist(),
                    "target_buoy": target_buoy.tolist(),
                    "trajectory_markers": trajectory_markers.tolist()
                })

            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to update camera server: {e}")


            # Normalize the observation using the loaded statistics
            obs_normalized = vec_env.normalize_obs(obs)

            # Get the action from the trained model
            action, _ = model.predict(obs_normalized, deterministic=True)
            # action, _ = model.predict(obs, deterministic=True)

            # Take the action in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Update the visualization
            env.unwrapped.step_sim()

            
            
            # Add a small delay to make the visualization viewable
            time.sleep(0.1)

            step_count += 1

            # Print current state (optional)
            print(
                f"Step {step_count}: Position (x={obs['x'][0]:.2f}, y={obs['y'][0]:.2f}, z={obs['z'][0]:.2f})"
            )
            print(f"Target position: {env.unwrapped.target_position}")
            print(f"Current reward: {reward:.2f}")

            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {step_count} steps")
                print(f"Total reward: {episode_reward:.2f}")
                break
    
    # Shut down the camera server before closing the environment
    stop_camera_server(camera_server_process)

    env.close()


def test_agent_manual_input():
    env = gym.make("BlueRov-v0", render_mode="human")

    episodes = 200

    for episode in range(episodes):
        obs, _ = env.reset()
        env.render()  # Initial render
        episode_reward = 0
        step_count = 0

        print(f"\nStarting Episode {episode + 1}")

        while True:
            if step_count < episodes / 2:
                action = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                action = np.array([0.0, 0.0, 1.0, 0.9])

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            env.unwrapped.step_sim()

            time.sleep(0.1)

            step_count += 1

            print(
                f"Step {step_count}: Position (x={obs['x'][0]:.2f}, y={obs['y'][0]:.2f}, z={obs['z'][0]:.2f})"
            )
            print(f"Current reward: {reward:.2f}")

            if terminated or truncated:
                print(f"Episode {episode + 1} finished after {step_count} steps")
                print(f"Total reward: {episode_reward:.2f}")
                break

    env.close()


def manual_control():
    """
    Test the environment with manual controls for debugging
    Keys:
    - W/S: Forward/Backward
    - A/D: Left/Right
    - Q/E: Rotate
    - R/F: Up/Down
    """
    env = gym.make("BlueRov-v0", render_mode="human")
    obs, _ = env.reset()
    env.render()

    while True:
        action = np.array([0.0, 0.0, 0.0, 0.0])

        key = input("Enter control (wasdqerf, x to exit): ").lower()

        if key == "x":
            break
        elif key == "w":
            action[0] = 1.0  # Forward
        elif key == "s":
            action[0] = -1.0  # Backward
        elif key == "a":
            action[1] = -1.0  # Left
        elif key == "d":
            action[1] = 1.0  # Right
        elif key == "q":
            action[3] = -1.0  # Rotate left
        elif key == "e":
            action[3] = 1.0  # Rotate right
        elif key == "r":
            action[2] = 1.0  # Up
        elif key == "f":
            action[2] = -1.0  # Down
        print(f"Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        env.unwrapped.step_sim()

        print(
            f"Position: x={obs['x'][0]:.2f}, y={obs['y'][0]:.2f}, z={obs['z'][0]:.2f}"
        )
        print(f"Reward: {reward:.2f}")

        if terminated or truncated:
            obs, _ = env.reset()
            print("Episode ended, resetting...")

    env.close()


if __name__ == "__main__":
    # Choose whether to run trained agent or manual control
    mode = input(
        "Enter mode (1 for trained agent, 2 for manual control and 3 for predefined manual input): "
    )

    if mode == "1":
        test_agent()
    elif mode == "2":
        manual_control()
    elif mode == "3":
        test_agent_manual_input()
    else:
        print("Invalid mode selected")
