from dataclasses import dataclass
import numpy as np
import open3d as o3d
import cv2
import torch

@dataclass
class CameraParameters:
    fx: float
    fy: float
    cx: float
    cy: float
    H: int
    W: int
    depth_scale: float = 1000.0

class BGRtoRGB(object):
    """bgr format to rgb"""

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

class DepthScale(object):
    """scale depth to meters"""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, depth):
        depth = depth.astype(np.float32)
        return depth * self.scale

class DepthFilter(object):
    """scale depth to meters"""

    def __init__(self, max_depth, min_depth=None):
        self.max_depth = max_depth
        self.min_depth = min_depth

    def __call__(self, depth):
        far_mask = depth > self.max_depth
        depth[far_mask] = 0.
        if self.min_depth is not None:
            near_mask = depth < self.min_depth
            depth[near_mask] = 0.
        return depth
    

def pointcloud_from_depth_torch(
    depth,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_type: str = "z",
    skip=1,
) -> np.ndarray:
    assert depth_type in ["z", "euclidean"], "Unexpected depth_type"

    rows, cols = depth.shape
    c, r = np.meshgrid(
        np.arange(cols, step=skip), np.arange(rows, step=skip), sparse=True)
    c = torch.from_numpy(c).to(depth.device)
    r = torch.from_numpy(r).to(depth.device)
    depth = depth[::skip, ::skip]
    valid = ~torch.isnan(depth)
    nan_tensor = torch.FloatTensor([float('nan')]).to(depth.device)
    z = torch.where(valid, depth, nan_tensor)
    x = torch.where(valid, z * (c - cx) / fx, nan_tensor)
    y = torch.where(valid, z * (r - cy) / fy, nan_tensor)
    pc = torch.dstack((x, y, z))

    if depth_type == "euclidean":
        norm = torch.linalg.norm(pc, axis=2)
        pc = pc * (z / norm)[:, :, None]
    return pc

# adapted from https://github.com/wkentaro/morefusion/blob/main/morefusion/geometry/estimate_pointcloud_normals.py
def estimate_pointcloud_normals(points):
    # These lookups denote yx offsets from the anchor point for 8 surrounding
    # directions from the anchor A depicted below.
    #  -----------
    # | 7 | 6 | 5 |
    #  -----------
    # | 0 | A | 4 |
    #  -----------
    # | 1 | 2 | 3 |
    #  -----------
    assert points.shape[2] == 3

    d = 2
    H, W = points.shape[:2]
    points = torch.nn.functional.pad(
        points,
        pad=(0, 0, d, d, d, d),
        mode="constant",
        value=float('nan'),
    )

    lookups = torch.tensor(
        [(-d, 0), (-d, d), (0, d), (d, d), (d, 0), (d, -d), (0, -d), (-d, -d)]
    ).to(points.device)

    j, i = torch.meshgrid(torch.arange(W), torch.arange(H))
    i = i.transpose(0, 1).to(points.device)
    j = j.transpose(0, 1).to(points.device)
    k = torch.arange(8).to(points.device)

    i1 = i + d
    j1 = j + d
    points1 = points[i1, j1]

    lookup = lookups[k]
    i2 = i1[None, :, :] + lookup[:, 0, None, None]
    j2 = j1[None, :, :] + lookup[:, 1, None, None]
    points2 = points[i2, j2]

    lookup = lookups[(k + 2) % 8]
    i3 = i1[None, :, :] + lookup[:, 0, None, None]
    j3 = j1[None, :, :] + lookup[:, 1, None, None]
    points3 = points[i3, j3]

    diff = torch.linalg.norm(points2 - points1, dim=3) + torch.linalg.norm(
        points3 - points1, dim=3
    )
    diff[torch.isnan(diff)] = float('inf')
    indices = torch.argmin(diff, dim=0)

    normals = torch.cross(
        points2[indices, i, j] - points1[i, j],
        points3[indices, i, j] - points1[i, j],
    )
    normals /= torch.linalg.norm(normals, dim=2, keepdims=True)
    return normals


############################################ Draw line mesh (copied from https://github.com/isl-org/Open3D/pull/738) ################################################
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
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
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
