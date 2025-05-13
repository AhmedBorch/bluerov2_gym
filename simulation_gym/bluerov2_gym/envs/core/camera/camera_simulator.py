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
        p.setGravity(0, 0, 0)  # Disable gravity
        self.debug = False
        # Load the ground plane
        #self.plane_id = p.loadURDF("plane.urdf")
        
        # Add colored balls
        
# Create colored spheres (balls)

        self.ball_ids = []
        self.ball_ids.append(self.create_colored_ball([3, 0, 0], [ 1, 0, 0]))  # Blue
        self.ball_ids.append(self.create_colored_ball([0, 5, 0.5], [1, 1, 0]))  # Yellow
        self.ball_ids.append(self.create_colored_ball([0, 5, -0.5], [0, 0, 1])) # Red
        #self.ball_ids.append(self.create_colored_ball([0, -3, 0], [0, 1, 0])) # Green
        for ball_id in self.ball_ids:
            p.resetBaseVelocity(ball_id, [0, 0, 0])  # Zero initial velocity
            p.resetBasePositionAndOrientation(ball_id, p.getBasePositionAndOrientation(ball_id)[0], [0, 0, 0, 1])  # Zero orientation
            p.resetBaseVelocity(ball_id, [0, 0, 0])  # Zero velocity
            if self.debug: print("Ball added at:", p.getBasePositionAndOrientation(ball_id))

        # Camera settings
        self.width = 640
        self.height = 480
        self.fov = 60
        self.aspect = self.width / self.height
        self.near = 0.1
        self.far = 100
        self.current_traj_leng = 0 #carefull in case you want to run multiple runs after each other
        
        if self.debug:print("All loaded bodies:")
        for i in range(p.getNumBodies()):
            if self.debug:print(f"Body {i}: {p.getBodyInfo(i)} at {p.getBasePositionAndOrientation(i)}")



    def create_colored_ball(self, position, color,sphere_radius=0.15):
        collision = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=color + [1])
        return p.createMultiBody(0, collision, visual, basePosition=position)

 
    def render_camera_view(self, position, orientation,new_buoy_positions=None,trajectory_markers=None):
        """
        Simulates a camera located at `position` with `orientation` (yaw, pitch, roll in radians).
        """

        if self.debug:print(f"[DEBUG] Camera position: {position}, orientation: {orientation}")
    
        # Convert Euler to quaternion

        # Updating the position
        #new_buoy_positions #e.g of that type [new_x, new_y, new_z]
        if new_buoy_positions is not None:
            new_buoy_positions = np.array(new_buoy_positions).flatten()
            assert len(new_buoy_positions) == 3, "Buoy pos must be of length 3"
            if self.debug:print(f"[DEBUG] Updating buoy position: {new_buoy_positions} {type(new_buoy_positions)}", flush=True)
            p.resetBasePositionAndOrientation(self.ball_ids[0], new_buoy_positions.tolist(), [0, 0, 0, 1])

        if trajectory_markers is not None:
            #print(f"[DEBUG] Raw trajectory_markers input: {trajectory_markers}", flush=True)
            
            trajectory_markers = np.array(trajectory_markers).flatten()
            #print(f"[DEBUG] Flattened trajectory_markers: {trajectory_markers}, length: {len(trajectory_markers)}", flush=True)

            while len(trajectory_markers) > self.current_traj_leng:
                # Since flattening gives a 1D array, we need to group every 3 values into an (x, y, z) point
                if (self.current_traj_leng + 1) * 3 > len(trajectory_markers):
                #     print(f"[WARNING] Incomplete point at index {self.current_traj_leng}", flush=True)
                    break

                index = self.current_traj_leng * 3
                pos = trajectory_markers[index:index+3]

                if len(pos) == 3:
                #     print(f"[DEBUG] Creating trajectory marker at: {pos.tolist()}", flush=True)
                    self.create_colored_ball(position=pos.tolist(), color=[0, 1, 0], sphere_radius=0.02)
                # else:
                #     print(f"[ERROR] Skipped invalid position data: {pos}", flush=True)

                self.current_traj_leng += 1

        q = p.getQuaternionFromEuler(orientation)

        # Compute forward-facing camera direction
        rot_matrix = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)
        camera_vector = rot_matrix @ np.array([0, 1, 0])  # forward #np.array([1, 0, 0])
        up_vector = rot_matrix @ np.array([0, 0, -1])      # up #np.array([0, 0, 1]) 

        # Set camera target to the robot's position
        camera_target = position + camera_vector
        view_matrix = p.computeViewMatrix(position, camera_target, up_vector)
        projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)

        # Step the simulation so the balls move (or update their positions)
        p.stepSimulation()

        # Get the camera image
        img_arr = p.getCameraImage(self.width, self.height, view_matrix, projection_matrix,renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.reshape(img_arr[2], (self.height, self.width, 4))[:, :, :3]
        rgb_array = rgb_array.astype(np.uint8)
        return rgb_array
