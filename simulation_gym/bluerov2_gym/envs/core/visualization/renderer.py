# renderer.py
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

class BlueRovRenderer:
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        if self.render_mode == "human":
            # Open the MeshCat visualizer once
            self.vis = meshcat.Visualizer().open()
            # Preload the entire scene
            self._setup_scene()
        else:
            self.vis = None

    def _setup_scene(self):
        # 1) Water surface (thin, semi-transparent)
        water_surface = g.Box([30, 30, 0.01])
        water_mat = g.MeshPhongMaterial(
            color=0x2389DA, opacity=0.3, transparent=True, side="DoubleSide"
        )
        self.vis["water_surface"].set_object(water_surface, water_mat)

        # 2) Water volume (deep, transparent)
        water_volume = g.Box([30, 30, -50])
        water_vol_mat = g.MeshPhongMaterial(
            color=0x1A6B9F, opacity=0.2, transparent=True
        )
        vol_tf = tf.translation_matrix([0, 0, -5])
        self.vis["water_volume"].set_object(water_volume, water_vol_mat)
        self.vis["water_volume"].set_transform(vol_tf)

        # 3) ROV vessel mesh
        vessel_mesh = g.DaeMeshGeometry.from_file(self._get_model_path())
        vessel_mat = g.MeshLambertMaterial(color=0x0000FF, wireframe=False)
        self.vis["vessel"].set_object(vessel_mesh, vessel_mat)

        # 4) Ground plane
        ground = g.Box([30, 30, 0.01])
        ground_mat = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_tf = tf.translation_matrix([0, 0, -10])
        self.vis["ground"].set_object(ground, ground_mat)
        self.vis["ground"].set_transform(ground_tf)

        # 5) Trail line (initially empty)
        self.trail_pts = []
        empty_pts = np.zeros((3, 0), dtype=np.float32)  # shape = (3, 0)
        trail_obj = g.LineSegments(
            g.PointsGeometry(position=empty_pts),
            g.LineBasicMaterial(color=0xFFFFFF)  # white
        )
        self.vis["trail"].set_object(trail_obj)

        # 6) Waypoint marker (small red cube)
        marker = g.Box([0.2, 0.2, 0.2])
        marker_mat = g.MeshPhongMaterial(color=0xFF0000)  # red
        self.vis["waypoint"].set_object(marker, marker_mat)
        self.vis["waypoint"].set_transform(np.eye(4))

    def _get_model_path(self):
        # Adjust this to point at your .dae model file
        return "/home/elex/aiproject/bluerov2_gym/simulation_gym/bluerov2_gym/assets/BlueRov2.dae"

    def reset(self):
        """Clear the trail at the start of each episode."""
        if not self.vis:
            return
        self.trail_pts = []
        empty_pts = np.zeros((3, 0), dtype=np.float32)
        trail_obj = g.LineSegments(
            g.PointsGeometry(position=empty_pts),
            g.LineBasicMaterial(color=0xFFFFFF)
        )
        self.vis["trail"].set_object(trail_obj)

    def render(self, *args, **kwargs):
        """No-op since scene is preloaded in __init__."""
        pass

    def step_sim(self, state):
        """Update vessel pose and extend the trail."""
        if not self.vis:
            return

        # Update vessel transform
        trans = tf.translation_matrix([state["x"], state["y"], state["z"]])
        rot = np.array([
            [np.cos(state["theta"]), -np.sin(state["theta"]), 0],
            [np.sin(state["theta"]),  np.cos(state["theta"]), 0],
            [0,                       0,                      1],
        ])
        M = np.eye(4)
        M[:3, :3] = rot
        M[:3, 3] = [state["x"], state["y"], state["z"]]
        self.vis["vessel"].set_transform(M)

        # Append to trail and redraw
        self.trail_pts.append([state["x"], state["y"], state["z"]])
        pts = np.array(self.trail_pts, dtype=np.float32).T  # shape = (3, N)
        trail_obj = g.LineSegments(
            g.PointsGeometry(position=pts),
            g.LineBasicMaterial(color=0xFFFFFF)
        )
        self.vis["trail"].set_object(trail_obj)

    def update_waypoint(self, goal):
        """Move the red cube marker to the new waypoint."""
        if not self.vis:
            return
        T = tf.translation_matrix([float(goal[0]), float(goal[1]), float(goal[2])])
        self.vis["waypoint"].set_transform(T)
