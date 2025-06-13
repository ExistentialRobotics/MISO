from typing import List
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import Dataset
from grid_opt.models.grid_atlas import GridAtlas
from .utils_geometry import pose_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def beautiful_rgb():
    color_list = [
        [1, 0.706, 0],
        [0, 0.651, 0.929],
        [0.17, 0.63, 0.17],  # Soft green
        [0.58, 0.40, 0.74],  # Vibrant purple
        [0.12, 0.65, 0.65],  # Teal
        [0.84, 0.15, 0.16],  # Rich red
    ]
    return color_list

def convert_to_colormap(v:np.ndarray, cmap_name='seismic', thresh=0.10) -> np.ndarray:
    """This function converts a numpy array of values to a colormap.

    Args:
        v: Input values to be converted to colors (shape (N,)).
        cmap_name (str, optional): Defaults to 'seismic'.

    Returns:
        np.ndarray: Output colors (shape (N, 3)) in RGB format.
    """
    v_min, v_max = -thresh, thresh
    abs_max = max(abs(v_min), abs(v_max))
    v_normalized = (v + abs_max) / (2 * abs_max)
    cmap = plt.get_cmap(cmap_name)  # blue-white-red
    colors = cmap(v_normalized)[:, :3]  # drop alpha channel
    return colors

def create_coordinate_frame(R, t, color=[0,0,1], size=0.5):
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    T = np.eye(4)
    T[:3,:3] = R 
    T[:3, 3] = t.flatten()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coord.transform(T)
    coord.paint_uniform_color(color)
    return coord

def create_lineset_from_numpy_traj(points_np:np.ndarray, color=[0,0,1]):
    """Given a list of 3D numpy vectors, create a corresponding lineset for visualization in open3d.
    """
    assert points_np.shape[1] == 3
    num_points = points_np.shape[0]
    # points_np = np.vstack(points)  # Shape (N, 3), where N is the number of points
    # Create lines to connect consecutive points
    lines = [[i, i + 1] for i in range(num_points - 1)]  # Line indices

    # Create the Open3D LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_np)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Optional: Add color to the lines
    colors = [color for _ in range(len(lines))]  
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def create_spheres_from_numpy_traj(points_np:np.ndarray, radius=0.08, color=[0,0,1]):
    spheres = []
    assert points_np.shape[1] == 3
    num_points = points_np.shape[0]
    for idx in range(num_points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)  # Small spheres
        sphere.translate(points_np[idx,:])
        sphere.paint_uniform_color(color)  
        spheres.append(sphere)
    return spheres

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


def visualize_submaps(
    grid_atlas: GridAtlas,
    dataset: Dataset,
    submap_ids: List[int],
    num_batches=5,
):
    pcd_colors = beautiful_rgb()
    pcd_list = []
    for submap_id in submap_ids:
        points = dataset.get_points_for_submap(submap_id, num_batches=num_batches)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color(pcd_colors[submap_id])
        pcd.estimate_normals()
        R_world_submap, t_world_submap = grid_atlas.updated_submap_pose(submap_id)
        T_world_submap = pose_matrix(R_world_submap, t_world_submap)
        pcd.transform(T_world_submap.detach().cpu().numpy())
        pcd_list.append(pcd)
    o3d.visualization.draw_geometries(pcd_list, point_show_normal=False)


def visualize_trajectories(
    traj_dict,
    traj_plot_path: str = None,
    plot_3d: bool = True,
    grid_on: bool = True,
    vis_now: bool = False,
    close_all: bool = True,
) -> None:
    from evo.core.trajectory import PosePath3D
    from evo.tools import plot as evoplot
    if close_all:
        plt.close("all")
    if plot_3d:
        plot_mode = evoplot.PlotMode.xyz
    else:
        plot_mode = evoplot.PlotMode.xy
    fig = plt.figure(f"Trajectory results")
    ax = evoplot.prepare_axis(fig, plot_mode)
    colors = beautiful_rgb()
    assert len(colors) >= len(traj_dict)  
    traj_idx = 0
    for label, poses in traj_dict.items():
        pose_path = PosePath3D(poses_se3=poses)
        evoplot.traj(
            ax=ax,
            plot_mode=plot_mode,
            traj=pose_path,
            label=label,
            color=colors[traj_idx],
            alpha=0.5,
        )
        traj_idx += 1

    plt.tight_layout()
    ax.legend(frameon=grid_on)
    if traj_plot_path is not None:
        plt.savefig(traj_plot_path, dpi=600)
    if vis_now:
        plt.show()
