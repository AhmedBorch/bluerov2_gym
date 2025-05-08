import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


file_path = os.path.join(os.path.dirname(__file__), "training_stats.pkl")
with open(file_path, "rb") as f:
    stats = pickle.load(f)

episode_rewards = stats["episode_rewards"]  # list of list of rewards per step
episode_lengths = stats["episode_lengths"]  # list of lengths per episode




total_rewards = [sum(rewards) for rewards in episode_rewards]



plt.figure(figsize=(15, 5))

# Plot total reward per episode
plt.subplot(1, 3, 1)
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.grid(True)
plt.legend()

# Plot episode lengths
plt.subplot(1, 3, 2)
plt.plot(episode_lengths)
plt.xlabel("Episode")
plt.ylabel("Episode Length (steps)")
plt.title("Episode Length per Episode")
plt.grid(True)

plt.subplot(1, 3, 3)

# Plot total reward per episode
# The color 

# Choose a colormap (e.g., 'viridis', 'plasma', 'cool', 'inferno')
cmap = plt.get_cmap('viridis')
N=len(episode_rewards)
# Create N colors spaced evenly across the colormap
colors = [cmap(i / (N - 1)) for i in range(N)]


for i, (reward, color) in enumerate(zip(episode_rewards, colors)):
    plt.plot(reward, color=color)

plt.xlabel("Step within Episode")
plt.ylabel("Reward at Step")
plt.ylim([-0.1,0.0])
plt.title("Reward per Step (Colored by Episode)")
plt.grid(True)

# Add a colorbar for episode number
norm = Normalize(vmin=1, vmax=N)
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Dummy array for the colorbar
cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02)
cbar.set_label("Episode Number")

plt.tight_layout()
plt.show()
