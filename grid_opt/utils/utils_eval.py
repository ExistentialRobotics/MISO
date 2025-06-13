import torch
import trimesh
import numpy as np
import open3d as o3d
from pytorch3d.ops import knn_points
import json
from evo.core.trajectory import PosePath3D
from evo.core import metrics
import grid_opt.utils.utils_geometry as utils_geometry
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def nn_correspondance(src_points, tgt_points, truncation=None, remove_far=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    src = torch.from_numpy(src_points).float().unsqueeze(0).to(device)  # (1, N, 3)
    tgt = torch.from_numpy(tgt_points).float().unsqueeze(0).to(device)  # (1, M, 3)

    knn = knn_points(src, tgt, K=1)

    dists = knn.dists.squeeze(-1)  # (1, N_src)
    idxs = knn.idx.squeeze(-1)     # (1, N_src)

    # Squared distance -> distance
    dists = torch.sqrt(dists)

    if truncation is not None and remove_far:
        mask = dists <= truncation
        dists = dists[mask]
        idxs = idxs[mask]

    dist_list = dists.cpu().numpy().tolist()
    nn_idx_list = idxs.cpu().numpy().tolist()

    return nn_idx_list, dist_list

def sample_points_from_mesh(mesh_file, mesh_sample_point=1000000, voxel_down_sample_res=0.02, input_format='mesh'):
    if input_format == 'mesh':
        mesh = trimesh.load(mesh_file, force='mesh')
        points, _ = trimesh.sample.sample_surface(mesh, count=mesh_sample_point)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    elif input_format == 'pointcloud':
        pcd = o3d.io.read_point_cloud(mesh_file)
    else:
        raise ValueError(f"Unknown input format {input_format}!")
    if voxel_down_sample_res > 0:
        pcd = pcd.voxel_down_sample(voxel_down_sample_res)
    points = np.asarray(pcd.points)
    return points

def filter_points_by_bound(points, bound):
    mask = (
        (points[:, 0] >= bound[0][0]) & (points[:, 0] <= bound[0][1]) &
        (points[:, 1] >= bound[1][0]) & (points[:, 1] <= bound[1][1]) &
        (points[:, 2] >= bound[2][0]) & (points[:, 2] <= bound[2][1])
    )
    return points[mask]

def filter_points_by_oriented_bound(points, obb):
    valid_indices = obb.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(points))
    num_valid = len(valid_indices)
    num_total = points.shape[0]
    logger.debug(f"{num_valid}/{num_total} points remained.")
    # o3d.visualization.draw_geometries([mesh,obb])
    return points[valid_indices,:]

def filter_points_by_gt_sdf(points, gt_sdf_func, min_sdf=-1e5, max_sdf=1e5):
    sdf_vals = gt_sdf_func(points)[:,None].astype(np.float32)  # N, 1
    sdf_vals = sdf_vals.flatten()
    mask_min = sdf_vals > min_sdf
    return points[mask_min,:]

def compute_chamfer_metrics(verts_pred, verts_trgt, 
                            threshold=0.01, 
                            truncation_acc=0.50, 
                            truncation_com=0.50):
    _, dist_p = nn_correspondance(verts_pred, verts_trgt, truncation_acc, True)  # Pred -> GT
    _, dist_r = nn_correspondance(verts_trgt, verts_pred, truncation_com, False) # GT -> Pred

    dist_p = np.array(dist_p)
    dist_r = np.array(dist_r)
    
    if len(dist_p) == 0:
        dist_p_mean = np.inf
    else:
        dist_p_mean = np.mean(dist_p)
    if len(dist_r) == 0:
        dist_r_mean = np.inf
    else:
        dist_r_mean = np.mean(dist_r)
    
    chamfer_l1 = 0.5 * (dist_p_mean + dist_r_mean)
    chamfer_l2 = np.sqrt(0.5 * (dist_p_mean + dist_r_mean))
    precision = np.mean((dist_p < threshold).astype('float')) * 100.0 if len(dist_p) > 0 else 0
    recall = np.mean((dist_r < threshold).astype('float')) * 100.0 if len(dist_r) > 0 else 0
    fscore = 2 * precision * recall / (precision + recall + 1e-8)
    
    metrics = {
        'MAE_accuracy (cm)': dist_p_mean * 100,
        'MAE_completeness (cm)': dist_r_mean * 100,
        'Chamfer_L1 (cm)': chamfer_l1 * 100,
        'Chamfer_L2 (cm)': chamfer_l2 * 100, 
        'Precision (%)': precision,
        'Recall (%)': recall,
        'F-score (%)': fscore
    }
    return metrics

def get_evo_trajectory(R, t) -> PosePath3D:
    """Convert pytorch rotation and translation tensors to a evo pose path.
    Input: 
        R: torch.tensor, shape (n, 3, 3)
        t: torch.tensor, shape (n, 3)
    Returns:
        PosePath3D:
    """
    pose_list = []
    n = R.shape[0]
    for var_id in range(n):
        Ti = utils_geometry.pose_matrix(R[var_id], t[var_id])
        pose_list.append(Ti.detach().cpu().numpy())
    return PosePath3D(poses_se3=pose_list)

def evo_trajectory_error(
    R1, t1,
    R2, t2,
    pose_relation = metrics.PoseRelation.translation_part,
    align: bool = True
) -> metrics.APE:
    """Align two dicts of 3D poses and evaluate the translation error

    Args:
        pose_dict1 (PoseDict): First set of poses
        pose_dict2 (PoseDict): Second set of poses
        pose_relation: evo translation_part, rotation_angle_deg, etc.
        align: if True, align the two trajectories as a pre-process step

    Returns:
        APE: metrics.APE object containing eval results
    """
    pose_path1 = get_evo_trajectory(R1, t1)
    pose_path2 = get_evo_trajectory(R2, t2)
    if align:
        pose_path2.align(pose_path1, correct_scale=False, correct_only_scale=False)
    ape_metric = metrics.APE(pose_relation)
    ape_data = (pose_path1, pose_path2)
    ape_metric.process_data(ape_data)
    return ape_metric

# Example
# if __name__ == "__main__":
#     file_pred = "results/pointsdf/learned_mesh.ply" 
#     file_trgt = "/home/sunghwan/data/Replica/room_0/mesh.ply"       

#     verts_pred = sample_points_from_mesh(file_pred, mesh_sample_point=1000000, voxel_down_sample_res=0.02)
#     verts_trgt = sample_points_from_mesh(file_trgt, mesh_sample_point=1000000, voxel_down_sample_res=0.02)

#     metrics = compute_chamfer_metrics(verts_pred, verts_trgt, threshold=0.05, truncation_acc=0.50, truncation_com=0.50)
#     formatted_metrics = {key: round(value, 6) for key, value in metrics.items()}

#     print(json.dumps(formatted_metrics, indent=4))
