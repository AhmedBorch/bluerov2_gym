import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np

import bluerov2_gym


class BlueRovRenderer:

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        self.vis = meshcat.Visualizer()
        self.vis.open()
        self.trail_markers = []  # For trajectory markers
        self.trail_positions = []
        self.step_counter = 0    # To track how many steps have passed


    def render(self, model_path):
        if self.render_mode != "human":
            return
        water_surface = g.Box([30, 30, 0.01])
        water_material = g.MeshPhongMaterial(
            color=0x2389DA, opacity=0.3, transparent=True, side="DoubleSide"
        )
        self.vis["water_surface"].set_object(water_surface, water_material)

        water_volume = g.Box([30, 30, -50])
        water_volume_material = g.MeshPhongMaterial(
            color=0x1A6B9F, opacity=0.2, transparent=True
        )
        water_volume_transform = tf.translation_matrix([0, 0, -5])
        self.vis["water_volume"].set_object(water_volume, water_volume_material)
        self.vis["water_volume"].set_transform(water_volume_transform)
        print("model_path: ", model_path)
        self.vis["vessel"].set_object(
            g.DaeMeshGeometry.from_file(model_path),
            g.MeshLambertMaterial(color=0x0000FF, wireframe=False),
        )

        ground = g.Box([30, 30, 0.01])
        ground_material = g.MeshPhongMaterial(color=0x808080, side="DoubleSide")
        ground_transform = tf.translation_matrix([0, 0, -10])
        self.vis["ground"].set_object(ground, ground_material)
        self.vis["ground"].set_transform(ground_transform)
        

    def get_robot_position(self):
        return [self.state["x"], self.state["y"], self.state["z"]]

    def plot_marker(self, position, orientation=None, marker_id=None):
        name = f"marker_{marker_id}" if marker_id else f"marker_{len(self.trail_markers)}"

        if orientation is not None:
            # Shaft of the arrow
            shaft_length = 0.04
            shaft_radius = 0.005
            shaft_geom = g.Cylinder(height=shaft_length, radius=shaft_radius)
            # Rotate cylinder from Z to X
            shaft_tf = tf.rotation_matrix(np.pi / 2, [0, 1, 0])
            # Translate so base is at origin and points along +X
            

            # Head of the arrow (sphere)
            head_radius = 0.01
            head_geom = g.Sphere(head_radius)
            head_tf = tf.translation_matrix([0,shaft_length/2, 0])


            # Add shaft and head as subpaths
            self.vis[name]["shaft"].set_object(shaft_geom, g.MeshLambertMaterial(color=0x00FF00))
            self.vis[name]["shaft"].set_transform(shaft_tf)

            self.vis[name]["head"].set_object(head_geom, g.MeshLambertMaterial(color=0x00FF00))
            self.vis[name]["head"].set_transform(head_tf)

            # Now apply yaw rotation (around Z) to the whole arrow
            rotation = tf.rotation_matrix(orientation, [0, 0, 1])
        else:
            # No orientation → just a sphere
            self.vis[name].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=0x00FF00))
            rotation = np.eye(4)

        # Final placement
        translation = tf.translation_matrix(position)
        final_tf = tf.concatenate_matrices(translation, rotation)
        self.vis[name].set_transform(final_tf)


        self.trail_markers.append(name)     


    def plot_target(self, target_position):
        target_sphere = g.Sphere(0.15)
        target_material = g.MeshLambertMaterial(color=0xFF0000)  # Bright red

        self.vis["target"].set_object(target_sphere, target_material)

        transform = tf.translation_matrix(target_position)
        self.vis["target"].set_transform(transform)


    def step_sim(self, state):
        self.state = state  # maybe wrong. check later
        if self.render_mode != "human":
            return

        translation = np.array([self.state["x"], self.state["y"], self.state["z"]])
        rotation_matrix = np.array(
            [
                [np.cos(self.state["theta"]), -np.sin(self.state["theta"]), 0],
                [np.sin(self.state["theta"]), np.cos(self.state["theta"]), 0],
                [0, 0, 1],
            ]
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        self.vis["vessel"].set_transform(transform_matrix)
        # Plot marker every 10 steps
        self.step_counter += 1
        if self.step_counter % 2 == 0:
            self.plot_marker(position=translation, orientation=self.state["theta"],marker_id=self.step_counter)