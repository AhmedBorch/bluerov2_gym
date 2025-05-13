import numpy as np

# Load the observations from file
observations = np.load("tlaloc/tested_observations_log.npy", allow_pickle=True)

# Show overall info
print("Type:", type(observations))
print("Shape:", observations.shape)
print("Data type:", observations.dtype)

# Display the first 5 observations
print("\nFirst 1 observations:")
print(observations[:1])

# If observations are arrays (not dicts), show stats per dimension
if isinstance(observations[0], (np.ndarray, list)):
    observations_array = np.array(observations.tolist())  # Ensure uniform shape
    print("\nStats per observation dimension:")
    for i in range(observations_array.shape[1]):
        print(f"Observation {i}: min={observations_array[:, i].min():.3f}, max={observations_array[:, i].max():.3f}, mean={observations_array[:, i].mean():.3f}")

# If observations are dicts (common in custom Gym envs), show keys and stats
elif isinstance(observations[0], dict):
    print("\nObservation keys:", observations[0].keys())
    for key in observations[0].keys():
        values = np.array([obs[key] for obs in observations])
        print(f"\nStats for '{key}':")
        print(f"  Shape: {values.shape}")
        print(f"  Min: {np.min(values):.3f}, Max: {np.max(values):.3f}, Mean: {np.mean(values):.3f}")
else:
    print("\nUnsupported observation format.")
