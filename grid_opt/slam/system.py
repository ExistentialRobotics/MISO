from os.path import join
import torch
from torch import Tensor
from grid_opt.models.grid_net import GridNet
from grid_opt.models.grid_atlas import GridAtlas
from grid_opt.datasets.submap_dataset import SubmapDataset
import grid_opt.utils.utils_geometry as utils_geometry
from .tracker import Tracker
from .mapper import Mapper
from .visualizer import Visualizer
from grid_opt.utils.utils_sdf import save_mesh
import logging
logger = logging.getLogger(__name__)


class System:
    """
    A class for the SLAM system that integrates tracking and mapping.
    """
    def __init__(
            self,
            model: GridAtlas,              
            dataset_track: SubmapDataset,
            dataset_map: SubmapDataset,
            cfg: dict,
            R_world_origin: Tensor = None,
            t_world_origin: Tensor = None,
            verbose=True,
        ):
        assert isinstance(model, GridAtlas), "Model must be an instance of GridAtlas."
        assert model.num_submaps == 0, "Input grid atlas is not empty."
        self.model = model
        self.cfg = cfg
        self.verbose = verbose
        self.dataset_track = dataset_track
        self.dataset_map = dataset_map
        self.max_replay_frames = cfg['mapping']['max_replay_frames']
        self.max_replay_freq = cfg['mapping']['max_replay_freq']
        self.init_odom = cfg['system']['init_odom']
        self.log_dir = cfg['system']['log_dir']
        self.initialize_system(Rws=R_world_origin, tws=t_world_origin)
        logger.info("System initialized with the following configuration:")
        logger.info(f"  - Verbose: {self.verbose}")
        logger.info(f"  - Initial odometry: {self.init_odom}")

    def currrent_submap(self) -> GridNet:
        """
        Return the current submap that tracking and mapping should be performed on.
        """
        return self.model.get_submap(self.model.curr_submap_id)
    
    def current_kf_id(self) -> int:
        """
        Return the latest keyframe ID that has been added.
        """
        return self.model.curr_kf_id
    
    def initialize_system(self, Rws: Tensor = None, tws: Tensor = None):
        """Initialize the system and the grid atlas.

        Args:
            Rws (Tensor, optional): Initial rotation in the world frame.
            tws (Tensor, optional): Initial translation in the world frame.
        """
        # Create the very first submap
        Rws = torch.eye(3) if Rws is None else Rws
        tws = torch.zeros(3,1) if tws is None else tws
        Rws = Rws.to(self.cfg['device'])
        tws = tws.to(self.cfg['device'])
        # Determine local bound of the new submap
        # max_bnd = torch.tensor(self.cfg['system']['submap_max_bound'])
        # points_frame = self.dataset_map.sampled_points_at_kf(0)
        # local_bound = utils_geometry.aabb_torch(points_frame, buffer=self.cfg['system']['submap_buffer'])
        # local_bound[:, 0] = torch.clamp(local_bound[:, 0], min=-max_bnd)
        # local_bound[:, 1] = torch.clamp(local_bound[:, 1], max= max_bnd)
        local_bound = torch.tensor(self.cfg['system']['submap_local_bound'])
        self.model.add_submap(local_bound, Rws, tws, self.cfg['system']['submap_size'])
        # self.model.add_submap(local_bound, torch.eye(3), torch.zeros(3,1), self.cfg['system']['submap_size'])
        # Create the very first keyframe
        self.model.add_kf(Rsk=torch.eye(3), tsk=torch.zeros(3,1))
        # self.model.add_kf(Rsk=Rws, tsk=tws)
        # Create tracker
        self.tracker = Tracker(
            model=self.currrent_submap(),
            dataset=self.dataset_track,
            cfg=self.cfg
        )
        # Create mapper
        self.mapper = Mapper(
            model=self.currrent_submap(),
            dataset=self.dataset_map,
            cfg=self.cfg
        )
        # Perform mapping to initialize the first submap
        self.mapper.mapping(mapping_kfs=[self.current_kf_id()], iterations=50, level_iterations=20)
        self.visualizer = Visualizer(self.model, cfg=self.cfg)

    def initialize_next_kf_in_submap(self):
        """
        Initialize the next keyframe in the current submap.
        """
        dst_id = self.current_kf_id() + 1
        src_id = dst_id - 1
        with torch.no_grad():
            R_src, t_src = self.model.updated_kf_pose_in_submap(src_id, submap_id=self.model.curr_submap_id)
            T_submap_src = utils_geometry.pose_matrix(R_src, t_src)
            if self.init_odom == 'external':
                T_src_dst = self.dataset_track.get_odometry_at_pose(src_id).to(T_submap_src)
            elif self.init_odom == 'static':
                T_src_dst = torch.eye(4).to(T_submap_src)
            else: 
                raise ValueError(f"Unknown odometry type: {self.init_odom}.")
            T_submap_dst = T_submap_src @ T_src_dst
            R_submap_dst = T_submap_dst[:3, :3]
            t_submap_dst = T_submap_dst[:3, [3]]
            self.model.add_kf(Rsk=R_submap_dst, tsk=t_submap_dst)
    
    def should_create_new_submap(self) -> bool:
        if self.model.num_keyframes_in_submap(self.model.curr_submap_id) >= self.cfg['system']['submap_size']:
            return True
        if self.tracker.latest_fov_overlap < self.cfg['system']['submap_fov_thresh']:
            return True
        return False
    
    def initialize_next_submap(self):
        dst_id = self.current_kf_id() + 1
        src_id = dst_id - 1
        with torch.no_grad():
            R_src, t_src = self.model.updated_kf_pose_in_world(src_id)
            T_world_src = utils_geometry.pose_matrix(R_src, t_src)
            T_src_dst = self.dataset_track.get_odometry_at_pose(src_id).to(T_world_src)
            T_world_dst = T_world_src @ T_src_dst
            R_world_dst = T_world_dst[:3, :3]
            t_world_dst = T_world_dst[:3, [3]]
        # Determine local bound of the new submap using the sampled points
        # max_bnd = torch.tensor(self.cfg['system']['submap_max_bound'])
        # points_frame = self.dataset_map.sampled_points_at_kf(dst_id)
        # local_bound = utils_geometry.aabb_torch(points_frame, buffer=self.cfg['system']['submap_buffer'])
        # local_bound[:, 0] = torch.clamp(local_bound[:, 0], min=-max_bnd)
        # local_bound[:, 1] = torch.clamp(local_bound[:, 1], max= max_bnd)
        local_bound = torch.tensor(self.cfg['system']['submap_local_bound'])
        # Create a new submap with world pose equal to the initialized keyframe pose
        # Note (yulun): since the keyframe pose is not optimized, this will cause 
        # misalignment between submaps.
        self.model.add_submap(local_bound, R_world_dst, t_world_dst, self.cfg['system']['submap_size'])
        # Add the next keyframe, which is also the anchor for the new submap
        # The local pose of the keyframe in the submap is identity
        kf_id = self.model.add_kf(Rsk=torch.eye(3), tsk=torch.zeros(3,1))
        assert kf_id == dst_id, "Keyframe ID mismatch."
        # Reset tracker
        self.tracker = Tracker(
            model=self.currrent_submap(),
            dataset=self.dataset_track,
            cfg=self.cfg
        )
        # Reset mapper
        self.mapper = Mapper(
            model=self.currrent_submap(),
            dataset=self.dataset_map,
            cfg=self.cfg
        )
        # Perform mapping to initialize the new submap
        self.mapper.mapping(mapping_kfs=[self.current_kf_id()], iterations=50, level_iterations=20)

    
    def run(self):
        first_frame_in_submap = 0
        assert self.current_kf_id() == 0, "No keyframe exists. Did you call initialize_system()?"
        # Incrementally train the frames in this submap
        while True:
            if self.model.num_keyframes == self.dataset_map.num_kfs:
                break
            if self.should_create_new_submap():
                # Save previous submap if requested
                if self.cfg['system']['save_submap_mesh']:
                    submap = self.currrent_submap()
                    mesh_path = join(self.log_dir, f'submap_{self.model.curr_submap_id}.ply')
                    save_mesh(submap, submap.bound, save_path=mesh_path, resolution=256)
                # Add the next frame in a new submap
                # Note (yulun): skip tracking for the first frame in the new submap
                self.initialize_next_submap()
                first_frame_in_submap = self.current_kf_id()
                continue
            else:
                # Add the next frame in the current submap
                self.initialize_next_kf_in_submap()
            head_kf = self.current_kf_id()
            self.tracker.track(optimize_kf=head_kf)
            # Mapping
            mapping_kfs = []   # Sample replay KFs to prevent forgetting
            replay_freq = (head_kf - first_frame_in_submap) // self.max_replay_frames
            replay_freq = max(replay_freq, self.max_replay_freq)
            for kf_id in range(first_frame_in_submap, head_kf, replay_freq):
                mapping_kfs.append(kf_id)
            mapping_kfs.append(head_kf)
            self.mapper.mapping(mapping_kfs=mapping_kfs, iterations=15, level_iterations=5)
            # Visualize
            frame_points = self.dataset_track.sampled_points_at_kf(head_kf)
            self.visualizer.set_current_frame_points(frame_points.detach().cpu().numpy())
            self.visualizer.update_geometries(stop_frame=head_kf+1)
            self.visualizer.update_view()
        self.visualizer.quit()