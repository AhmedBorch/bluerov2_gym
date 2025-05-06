import pybullet as p
import pybullet_data
import numpy as np
import time

class CameraSimulator:
    def __init__(self):
        p.connect(p.GUI) # Direct mode (no rendering)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Add this line before adding objects to the simulation
        p.resetSimulation()

        # Load the ground plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Add colored balls
        
# Create colored spheres (balls)

        self.ball_ids = []
        self.ball_ids.append(self.create_colored_ball([3, 0, 0], [0, 0, 1]))  # Blue
        self.ball_ids.append(self.create_colored_ball([0, 3, 0], [1, 1, 0]))  # Yellow
        self.ball_ids.append(self.create_colored_ball([-3, 0, 0], [1, 0, 0])) # Red
        self.ball_ids.append(self.create_colored_ball([0, -3, 0], [0, 1, 0])) # Green
        for ball_id in self.ball_ids:
            print("Ball added at:", p.getBasePositionAndOrientation(ball_id))


        # Camera settings
        self.width = 640
        self.height = 480
        self.fov = 60
        self.aspect = self.width / self.height
        self.near = 0.1
        self.far = 100
        
        print("All loaded bodies:")
        for i in range(p.getNumBodies()):
            print(f"Body {i}: {p.getBodyInfo(i)} at {p.getBasePositionAndOrientation(i)}")



    def create_colored_ball(self, position, color):
        sphere_radius = 0.5
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=color + [1])
        return p.createMultiBody(1, collision, visual, basePosition=position)

 
    def render_camera_view(self, position, orientation):
        """
        Simulates a camera located at `position` with `orientation` (yaw, pitch, roll in radians).
        """
        # Convert Euler to quaternion
        q = p.getQuaternionFromEuler(orientation)

        # Compute forward-facing camera direction
        rot_matrix = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        camera_vector = rot_matrix @ np.array([1, 0, 0])  # forward
        up_vector = rot_matrix @ np.array([0, 0, 1])      # up

        # Set camera target to the robot's position
        camera_target = position + camera_vector
        view_matrix = p.computeViewMatrix(position, camera_target, up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

        # Step the simulation so the balls move (or update their positions)
        p.stepSimulation()

        # Get the camera image
        img_arr = p.getCameraImage(self.width, self.height, view_matrix, projection_matrix)
        rgb_array = np.reshape(img_arr[2], (self.height, self.width, 4))[:, :, :3]
        rgb_array = rgb_array.astype(np.uint8)
        return rgb_array
