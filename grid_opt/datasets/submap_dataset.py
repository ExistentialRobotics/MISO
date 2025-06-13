import torch
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)

class SubmapDataset(Dataset):
    """
    A base class for SLAM datasets utilizing multiple submaps.
    """
    def __init__(self):
        super().__init__()

    @property
    def num_kfs(self) -> int:
        """
        Get the number of frames in the dataset.
        """
        raise NotImplementedError
    
    def get_odometry_at_pose(self, src_id) -> Tensor:
        """
        Get the odometry estimates from src_id to src_id + 1.
        Returning the 4x4 transformation matrix in torch
        """
        raise NotImplementedError
    
    def sampled_points_at_kf(self, kf_id) -> Tensor:
        """
        Get the sampled points at the keyframe as a (N,3) tensor.
        Returned in the local reference frame.
        """
        raise NotImplementedError
    
    def select_keyframes(self, kf_ids):
        """Downselect to the specified keyframes.
        Only the specified keyframes will be used within __getitem__.
        """
        raise NotImplementedError

    def unselect_keyframes(self):
        """Undo any previous keyframe selection.
        """
        raise NotImplementedError
    
    def true_kf_pose_in_world(self, kf_id):
        """
        Get the true pose of the keyframe in the world frame.
        Returning (R, t) as torch tensor
        """
        raise NotImplementedError
    
    def noisy_kf_pose_in_world(self, kf_id):
        raise NotImplementedError
    
    def __getitem__(self, index):
        """This function should be implemented by the derived class.
        It should return two dictionaries as below.
        input_dict = {
            'coords_frame': None,
            'sample_frame_ids': None,
            'weights': None
        }
        gt_dict = {
            'sdf': None,
            'sdf_valid': None,
            'sdf_signs': None,
        }
        NOTE: 
        - The 'sdf_signs' field should be a tensor of shape (N, 1) with values:
            -1 in occupied space
             0 ON AND NEAR surface (within truncation distance)
             1 in free space
        """
        raise NotImplementedError





