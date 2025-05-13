import numpy as np
import matplotlib.pyplot as plt

# Load the actions
actions = np.load("tlaloc/tested_actions_log.npy")

# Define labels for each action dimension
labels = ["thrust_x", "thrust_y", "thrust_z", "yaw_rate"]

# Create a figure with 4 vertically stacked subplots
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

for i in range(actions.shape[1]):
    ax = axes[i]
    ax.plot(actions[:, i])
    ax.set_ylabel(labels[i])
    ax.grid(True)
    ax.set_title(f"Action {i}: {labels[i]}")

axes[-1].set_xlabel("Timestep")

plt.tight_layout()
plt.show()
