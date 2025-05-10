import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


file_path = os.path.join(os.path.dirname(__file__), "200000_trained_network/training_stats.pkl")
with open(file_path, "rb") as f:
    stats = pickle.load(f)

episode_rewards = stats["episode_rewards"]  # list of list of rewards per step
episode_lengths = stats["episode_lengths"]  # list of lengths per episode




total_rewards = [sum(rewards) for rewards in episode_rewards]



# Plotting
plt.figure(figsize=(18, 5))

# Plot total reward per episode
plt.subplot(1, 3, 1)
plt.plot(total_rewards)
plt.xlabel("Episode", fontsize=17)  # Increased fontsize
plt.ylabel("Total Reward", fontsize=17)  # Increased fontsize
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)  # Increased tick number size

# Plot episode lengths
plt.subplot(1, 3, 2)
plt.plot(episode_lengths, color="C3")
plt.xlabel("Episode", fontsize=17)  # Increased fontsize
plt.ylabel("Episode Length (steps)", fontsize=17)  # Increased fontsize
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)  # Increased tick number size

# Plot reward curves per episode with color scale
plt.subplot(1, 3, 3)
cmap = plt.get_cmap('viridis')
N = len(episode_rewards)
colors = [cmap(i / (N - 1)) for i in range(N)]

for i, (reward, color) in enumerate(zip(episode_rewards, colors)):
    plt.plot(reward, color=color)

plt.xlabel("Step within Episode", fontsize=17)  # Increased fontsize
plt.ylabel("Reward at Step", fontsize=17)  # Increased fontsize
plt.ylim([0,-0.3])
plt.grid(True)
plt.tick_params(axis='both', labelsize=14)  # Increased tick number size

# Add a colorbar for episode number
norm = Normalize(vmin=1, vmax=N)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for the colorbar
cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
cbar.set_label("Episode Number", fontsize=17)  # Increased fontsize

# Remove title
plt.tight_layout()
plt.show()
