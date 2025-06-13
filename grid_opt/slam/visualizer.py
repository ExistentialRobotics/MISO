import numpy as np
import torch
import open3d as o3d
from grid_opt.models.grid_atlas import GridAtlas
import grid_opt.utils.utils_vis as utils_vis
import grid_opt.utils.utils_sdf as utils_sdf
import grid_opt.utils.utils_geometry as utils_geometry
import logging
logger = logging.getLogger(__name__)

class Visualizer:
    """
    A class for visualizing SLAM process using Open3D.
    """
    def __init__(self, grid_atlas: GridAtlas, cfg):
        assert isinstance(grid_atlas, GridAtlas), "Model must be an instance of GridAtlas."
        self.model = grid_atlas
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.cfg = cfg
        self.enable = cfg['visualizer']['enable']
        self.mesh_vis_freq = cfg['visualizer']['mesh_vis_freq']
        self.show_mesh = cfg['visualizer']['show_mesh']
        self.show_path = cfg['visualizer']['show_path']
        self.show_curr_pose = cfg['visualizer']['show_curr_pose']
        self.show_submap_obb = cfg['visualizer']['show_submap_obb']
        self.show_submap_pcd = cfg['visualizer']['show_submap_pcd']
        logger.info("Initializing visualizer.")
        logger.info(f"  - enable: {self.enable}.")
        logger.info(f"  - mesh_vis_freq: {self.mesh_vis_freq}.")
        logger.info(f"  - show_mesh: {self.show_mesh}.")
        logger.info(f"  - show_path: {self.show_path}.")
        logger.info(f"  - show_curr_pose: {self.show_curr_pose}.")
        logger.info(f"  - show_submap_obb: {self.show_submap_obb}.")
        logger.info(f"  - show_submap_pcd: {self.show_submap_pcd}.")
        self._initialize_geometries()
        self._initialize_visualizer()

    def _initialize_geometries(self):
        self.mesh = None
        self.pcd_list = []
        self.mesh_list = []
        self.obb_list = []
        self.path = None
        self.curr_pose = None
        self.curr_points_np = None
        self.curr_pcd = None

    def _initialize_visualizer(self):
        if not self.enable: return
        w_name = "SLAM Visualizer"
        self.vis.create_window(
            window_name=w_name, width=2560, height=1600
        )  # 1920, 1080
        self.vis.get_render_option().line_width = 500
        self.vis.get_render_option().light_on = True
        self.vis.get_render_option().point_size = 0.2
        self.vis.get_render_option().mesh_shade_option = (
            o3d.visualization.MeshShadeOption.Color
        )

    def set_current_frame_points(self, points_np: np.ndarray):
        self.curr_points_np = points_np
    
    def update_geometries(self, stop_frame):
        if not self.enable: return
        if self.show_mesh: 
            # Further throttle mesh update rate
            mesh = (stop_frame) % self.mesh_vis_freq == 0
        else:
            mesh = False
        if self.show_submap_pcd:
            # Further throttle mesh update rate
            submap_pcd = (stop_frame) % self.mesh_vis_freq == 0
        else:
            submap_pcd = False
        # Trajectory
        if self.show_path:
            if self.path is not None:
                self.vis.remove_geometry(self.path)
            t_est = torch.zeros((stop_frame, 3, 1))
            for frame_id in range(stop_frame):
                Rwf_est, twf_est = self.model.updated_kf_pose_in_world(frame_id)
                t_est[frame_id] = twf_est.detach().cpu()
            self.path = utils_vis.create_lineset_from_numpy_traj(t_est.squeeze().detach().cpu().numpy(), color=[0, 0, 1])
            self.vis.add_geometry(self.path)
        # Current pose
        if self.show_curr_pose:
            if self.curr_pose is not None:
                self.vis.remove_geometry(self.curr_pose)
            Rwf_est, twf_est = self.model.updated_kf_pose_in_world(stop_frame - 1)
            T = utils_geometry.pose_matrix(Rwf_est, twf_est).detach().cpu().numpy()
            new_pose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            new_pose.transform(T)
            self.curr_pose = new_pose   
            self.vis.add_geometry(self.curr_pose)
            # Also show current observed PCD
            if self.curr_pcd is not None:
                self.vis.remove_geometry(self.curr_pcd)
            # Create a point cloud from the current frame points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.curr_points_np)
            pcd.paint_uniform_color([1, 0, 0])
            pcd.transform(T)
            self.curr_pcd = pcd
            self.vis.add_geometry(self.curr_pcd)
        # Global mesh
        if mesh:
            if self.mesh is not None:
                self.vis.remove_geometry(self.mesh)
            mesh_o3d = utils_sdf.save_mesh(self.model, self.model.global_bound(), save_path=None, resolution=128)
            self.mesh = mesh_o3d
            self.vis.add_geometry(self.mesh)
        # Submap PCD
        if submap_pcd:
            # Update the mesh for the current submap
            submap_id = self.model.curr_submap_id
            submap = self.model.get_submap(submap_id)
            mesh_o3d = utils_sdf.save_mesh(submap, submap.bound, save_path=None, resolution=256)
            # Transform mesh to world frame
            Rwf_est, twf_est = self.model.updated_submap_pose(submap_id)
            T = utils_geometry.pose_matrix(Rwf_est, twf_est).detach().cpu().numpy()
            mesh_o3d = mesh_o3d.transform(T)
            pcd_o3d = mesh_o3d.sample_points_uniformly(number_of_points=100000)
            colors = utils_vis.beautiful_rgb()
            color_submap = np.array(colors[submap_id % len(colors)])
            mesh_o3d.paint_uniform_color(color_submap)
            pcd_o3d.paint_uniform_color(color_submap)
            if submap_id > len(self.pcd_list) - 1:
                # new submap to visualize
                self.pcd_list.append(pcd_o3d)
                self.mesh_list.append(mesh_o3d)
            else:
                # update the existing submap
                self.vis.remove_geometry(self.pcd_list[submap_id])
                self.pcd_list[submap_id] = pcd_o3d
                self.mesh_list[submap_id] = mesh_o3d
            self.vis.add_geometry(self.pcd_list[submap_id])
        # Submap oriented bounding box 
        if self.show_submap_obb:
            submap_id = self.model.curr_submap_id
            colors = utils_vis.beautiful_rgb()
            color_submap = np.array(colors[submap_id % len(colors)])
            new_obb = self.model.submap_obb_in_world(submap_id, color=color_submap)
            if submap_id <= len(self.pcd_list) - 1:
                self.vis.remove_geometry(self.obb_list[submap_id])
                self.obb_list[submap_id] = new_obb
            else:
                self.obb_list.append(new_obb)
            self.vis.add_geometry(self.obb_list[submap_id])            

    def update_view(self):
        if not self.enable: return
        self.vis.poll_events()
        self.vis.update_renderer()

    def quit(self):
        if not self.enable: return
        self.vis.destroy_window()