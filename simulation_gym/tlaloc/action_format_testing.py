import numpy as np

# Load the actions from file
actions = np.load("tlaloc/tested_actions_log.npy")

# Show overall info
print("Type:", type(actions))
print("Shape:", actions.shape)
print("Data type:", actions.dtype)

# Display the first 5 actions
print("\nFirst 5 actions:")
print(actions[:5])

# Display min, max, mean of each action dimension (e.g., thrust_x, etc.)
print("\nStats per action dimension:")
for i in range(actions.shape[1]):
    print(f"Action {i}: min={actions[:, i].min():.3f}, max={actions[:, i].max():.3f}, mean={actions[:, i].mean():.3f}")
