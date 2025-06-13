import argparse
from os.path import join
import numpy as np
from grid_opt.configs import *
from grid_opt.utils.utils_sdf import *
from grid_opt.slam.fuser import Fuser
from grid_opt.slam.system import System
from grid_opt.datasets.submap_dataset import SubmapDataset
import grid_opt.utils.utils_vis as utils_vis
import grid_opt.utils.utils_sdf as utils_sdf
import grid_opt.utils.utils_eval as utils_eval
import grid_opt.utils.utils_scannet as utils_scannet
from evo.core import metrics as evo_metrics
import open3d as o3d
import json
from math import radians
import logging
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, help='Path to config file.', default='./configs/lidar/ncd_quad.yaml')
parser.add_argument('--config', type=str, help='Path to config file.', default='./configs/rgbd/scannet.yaml')
parser.add_argument('--default_config', type=str, help='Path to config file.', default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/demo/mapping')
parser.add_argument('--pose_init', type=str, default='gt')  # reg_icp OR kiss_icp OR gt
parser.add_argument('--scannet_root', type=str, default='./data/ScanNet/scans')
parser.add_argument('--scene', type=str, default='0000_00')
parser.add_argument(
    '--feature_levels',
    nargs='*',
    default=[0, 1],
    type=int)
parser.add_argument('--use_sdf', action='store_true', 
                    help='Skip the fine-tuning with SDF.')


def create_configs_scannet(args, dataset: SubmapDataset):
    cfg = load_config(args.config, args.default_config)
    # Model settings
    scene = utils_scannet.scannet_scenes()[args.scene]
    cfg['model']['grid']['bound'] = scene.bound
    cfg['model']['pose']['num_poses'] = dataset.num_kfs
    cfg['model']['pose']['optimize'] = False
    # Mapping settings
    cfg['mapping']['weight_sdf'] = 1.0
    cfg['mapping']['weight_eik'] = 0.0
    cfg['mapping']['weight_fs'] = 0.1
    cfg['mapping']['learning_rate'] = 1e-3
    cfg['mapping']['verbose'] = True
    # System setting
    cfg['system']['log_dir'] = join(args.save_dir, "system")
    cfg['train']['log_dir'] = join(args.save_dir, "train")
    cfg['train']['grid_training_mode'] = 'joint'
    # alignment settings
    cfg['align']['latent_levels'] = args.feature_levels
    cfg['align']['skip_finetune'] = not args.use_sdf
    # print("################ DEBUG: Using feature levels:", cfg['align']['latent_levels'])
    # print("################ DEBUG: Use sdf:", args.use_sdf)
    # print("################ DEBUG: Skip fine-tune:", cfg['align']['skip_finetune'])
    return cfg


def initialize_scannet(args):
    # With this code, we only do a 'dry run' over the dataset sequence
    # in order to create the submap structure
    dataset = utils_scannet.create_scannet_dataset(
        args.scannet_root, args.scene, n_rays=200,
        frame_downsample=1)
    cfg = load_config(args.config, args.default_config)
    cfg = create_configs_scannet(args, dataset)
    # Disable incremental tracking, mapping, and vis
    cfg['tracking']['disable'] = True
    cfg['tracking']['verbose'] = False
    cfg['mapping']['disable'] = True
    cfg['mapping']['verbose'] = False
    cfg['visualizer']['enable'] = False
    grid_atlas = GridAtlas(cfg['model'], device=cfg['device'], dtype=torch.float32) 
    grid_atlas.to(cfg['device'])
    R_world_origin, t_world_origin = dataset.true_kf_pose_in_world(0)
    system = System(
        model=grid_atlas,
        dataset_track=dataset,
        dataset_map=dataset,
        cfg=cfg,
        R_world_origin=R_world_origin,
        t_world_origin=t_world_origin,
        verbose=False
    )
    system.run()  
    return cfg, grid_atlas, dataset


def visualize_grid_atlas(grid_atlas:GridAtlas, save_dir=None, postfix=None,
                         save_submaps=False, window_name="Submaps in Global Frame"):
    if save_dir is not None:
        utils.cond_mkdir(save_dir)
        save_filename = join(save_dir, f'fused_mesh_{postfix}.ply') if postfix is not None else join(save_dir, 'fused_mesh.ply')
        utils_sdf.save_mesh(grid_atlas, grid_atlas.global_bound(), save_path=save_filename)
    meshes = []
    for i in range(grid_atlas.num_submaps):
        submap = grid_atlas.get_submap(i)
        R, t = grid_atlas.updated_submap_pose(i)
        T = utils_geometry.pose_matrix(R, t)
        
        mesh_path = None
        if save_dir is not None and save_submaps:
            mesh_path = join(save_dir, f'submap_{i}.ply')
        mesh = utils_sdf.save_mesh(submap, submap.bound, transform=T, save_path=mesh_path)
        meshes.append(mesh)
    o3d.visualization.draw_geometries(meshes, window_name=window_name)


def run_alignment(cfg, grid_atlas, dataset):
    fuser = Fuser(
        model=grid_atlas,
        dataset=dataset,
        cfg=cfg
    )
    align_info = fuser.align()
    return align_info


def evaluate_alignment_error(grid_atlas, dataset):
    R_true = utils_geometry.identity_rotations(grid_atlas.num_submaps)
    t_true = torch.zeros((grid_atlas.num_submaps, 3, 1))
    R_sol = utils_geometry.identity_rotations(grid_atlas.num_submaps)
    t_sol = torch.zeros((grid_atlas.num_submaps, 3, 1))
    for submap_id in range(grid_atlas.num_submaps):
        anchor_kf = grid_atlas.anchor_kf_for_submap(submap_id)
        Rws, tws = grid_atlas.updated_submap_pose(submap_id)
        Rws_true, tws_true = dataset.true_kf_pose_in_world(anchor_kf)
        R_sol[submap_id] = Rws
        t_sol[submap_id] = tws
        R_true[submap_id] = Rws_true
        t_true[submap_id] = tws_true
    metrics_t = utils_eval.evo_trajectory_error(R_true, t_true, R_sol, t_sol, align=True, 
                                                pose_relation=evo_metrics.PoseRelation.translation_part).get_all_statistics()
    metrics_R = utils_eval.evo_trajectory_error(R_true, t_true, R_sol, t_sol, align=True,
                                                pose_relation=evo_metrics.PoseRelation.rotation_part).get_all_statistics()
    metrics = {
        'rmse_tran (cm)': 100 * metrics_t['rmse'],
        'rmse_deg': utils_geometry.chordal_to_degree(metrics_R['rmse']),
    }
    return metrics


def construct_submap_point_clouds(grid_atlas:GridAtlas, dataset:SubmapDataset):
    """
    Construct a point cloud for each submap in the grid atlas.
    The point clouds are represented in the submap's local frames.
    """
    submap_points = [[] for _ in range(grid_atlas.num_submaps)]
    for _ in range(3):
        input_dict, gt_dict = dataset[0]  # Each call will randomly sample observed coordinates
        coords_frame = input_dict['coords_frame'].detach().cpu()
        sample_frame_ids = input_dict['sample_frame_ids'][:,0].detach().cpu()
        sdfs = gt_dict['sdf'].detach().cpu()
        # Keep only surface points
        surface_mask = torch.abs(sdfs) < 1e-4
        valid_indices = torch.nonzero(surface_mask, as_tuple=False)[:, 0]
        coords_frame = coords_frame[valid_indices, :]
        sample_frame_ids = sample_frame_ids[valid_indices]
        unique_frame_ids = np.unique(sample_frame_ids.detach().cpu().numpy()).tolist()
        for frame_id in unique_frame_ids:
            submap_id = grid_atlas.submap_id_for_kf(frame_id)
            anchor_frame = grid_atlas.anchor_kf_for_submap(submap_id)
            idxs_select = torch.nonzero(sample_frame_ids == frame_id, as_tuple=False).squeeze(1)
            points_frame = coords_frame[idxs_select, :]
            R_world_frame, t_world_frame = dataset.true_kf_pose_in_world(frame_id)
            R_world_submap, t_world_submap = dataset.true_kf_pose_in_world(anchor_frame)
            points_world = utils_geometry.transform_points_to(
                points_frame, R_world_frame, t_world_frame
            )
            points_submap = utils_geometry.transfrom_points_from(
                points_world, R_world_submap, t_world_submap
            )
            submap_points[submap_id].append(points_submap)
    submap_points = [torch.cat(points, dim=0) for points in submap_points]
    print(f"Constructed point clouds for {len(submap_points)} submaps.")
    for i, points in enumerate(submap_points):
        print(f"Submap {i}: {points.shape} points.")
    return submap_points


def visualize_submap_point_clouds_iterations(iteration_results, submap_points, center, save_dir=None):
    cond_mkdir(save_dir)
    radius = 1.5
    height = 1.5
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)  # offscreen rendering
    # Configure the view control
    ctr = vis.get_view_control()
    render_opt = vis.get_render_option()
    render_opt.background_color = [1, 1, 1]  # white background
    render_opt.point_size = 0.3
    angle = radians(45)
    cam_pos = center + radius * np.array([np.cos(angle), np.sin(angle), height])
    up = np.array([0, 0, 1])  # z-axis up
    ctr.set_lookat(center)
    ctr.set_front(cam_pos - center)
    ctr.set_up(up)
    ctr.set_zoom(0.5)
    # Create the geometry list
    geometry_list = []
    transform_list = []
    num_submaps = len(submap_points)
    submap_colors = utils_vis.beautiful_rgb()
    for submap_id in range(num_submaps):
        points = submap_points[submap_id]
        assert points.shape[1] == 3, f"Expected points shape (N, 3), got {points.shape}"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.numpy())
        pcd.paint_uniform_color(submap_colors[submap_id])
        geometry_list.append(pcd)
        transform_list.append(np.eye(4))
    # Animate!
    for iter_idx, iter_poses in iteration_results.items():
        for submap_id in range(len(submap_points)):
            points = submap_points[submap_id]
            T_prev = transform_list[submap_id]
            T_curr = iter_poses[submap_id].detach().cpu().numpy()
            assert T_curr.shape == (4, 4), f"Expected T shape (4, 4), got {T_curr.shape}"
            geometry_list[submap_id].transform(np.linalg.inv(T_prev))  # Reset to identity pose
            geometry_list[submap_id].transform(T_curr)
            transform_list[submap_id] = T_curr
            if iter_idx == 0:
                vis.add_geometry(geometry_list[submap_id])
            else:
                vis.update_geometry(geometry_list[submap_id])
        vis.poll_events()
        vis.update_renderer()
        # time.sleep(0.2)  # Pause for a moment to visualize the change
        if save_dir is not None:
            frame_path = os.path.join(save_dir, f"frame_{iter_idx:03d}.png")
            vis.capture_screen_image(frame_path)
    vis.destroy_window()


def main_scannet():
    np.random.seed(55)
    torch.manual_seed(55)
    args = parser.parse_args()
    model_path = join(args.save_dir, 'grid_atlas.pth')
    cfg, grid_atlas, dataset = initialize_scannet(args)

    exp_name = ''
    if args.feature_levels == [0] and not args.use_sdf:
        exp_name = 'use_coarse_features'
    elif args.feature_levels == [0, 1] and not args.use_sdf:
        exp_name = 'use_fine_features'
    elif args.use_sdf and args.feature_levels == []:
        exp_name = 'use_sdf_only'
    elif args.use_sdf and args.feature_levels == [0, 1]:
        exp_name = 'full'
    else:
        exp_name = 'unknown_setting'

    save_path = join(args.save_dir, 'submap_alignment', exp_name)
    utils.cond_mkdir(save_path)

    # Load the model from the saved path
    grid_atlas = torch.load(model_path)
    
    # For this demo, we will manually perturb the submap poses in the world frame
    # By default, we fix anchor first submap
    noise_rot = utils_geometry.wrapped_gaussian_rotations(grid_atlas.num_submaps, std_rad=radians(10.0)).to(grid_atlas.device)
    noise_tra = utils_geometry.gaussian_translations(grid_atlas.num_submaps, stddev=0.50).to(grid_atlas.device)
    for i in range(1, grid_atlas.num_submaps):
        R, t = grid_atlas.initial_submap_pose(i)
        R_noisy = R @ noise_rot[i]
        t_noisy = t + noise_tra[i].reshape(3, 1)
        grid_atlas.set_submap_pose(i, R_noisy, t_noisy)
    
    # Before alignment
    # mesh_save_dir = join(save_path, 'submaps')
    visualize_grid_atlas(grid_atlas, save_dir=save_path, postfix='before_alignment',
                         window_name="Submaps before alignment")
    metrics_bef = evaluate_alignment_error(grid_atlas, dataset)

    # Run submap alignment
    align_info = run_alignment(cfg, grid_atlas, dataset)
    
    # After alignment
    visualize_grid_atlas(grid_atlas, save_dir=save_path, postfix='after_alignment',
                         window_name="Submaps after alignment")
    metrics_aft = evaluate_alignment_error(grid_atlas, dataset)

    # Print results
    print("Before alignment metrics:\n", json.dumps(metrics_bef, indent=4))
    print("After alignment metrics:\n",  json.dumps(metrics_aft, indent=4))
    # save the metrics to a json file
    metrics_save_path = join(save_path, 'alignment_result.json')
    with open(metrics_save_path, 'w') as f:
        json.dump({
            'before_alignment': metrics_bef,
            'after_alignment': metrics_aft
        }, f, indent=4)

    # Visualize iterations
    submap_points = construct_submap_point_clouds(grid_atlas, dataset)

    if args.feature_levels == [0] and not args.use_sdf:
        iteration_results = align_info['hier_latent_level0_L2']['iteration_results']
    elif args.feature_levels == [0, 1] and not args.use_sdf:
        iteration_results = align_info['hier_latent_level1_L2']['iteration_results']
    elif args.use_sdf:
        iteration_results = align_info['hier_sdf_L2']['iteration_results']
    else:
        return
    
    scene_obb = dataset.compute_scene_obb()
    center = scene_obb.get_center()
    visualize_submap_point_clouds_iterations(iteration_results, submap_points, center, join(save_path, 'submap_iterations'))
    # To convert the generated screenshots to a gif, in command line:
    # convert -delay 10 -loop 0 *.png myimage.gif
    

if __name__ == "__main__":
    main_scannet()
