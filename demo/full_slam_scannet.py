import argparse
from os.path import join
import json
import pandas as pd
from copy import deepcopy
import numpy as np
import open3d as o3d
import trimesh
import pysdf
from evo.core import metrics as evo_metrics
from grid_opt.configs import *
from grid_opt.utils.utils_sdf import *
from grid_opt.models.encoder import Encoder, EncoderObservation
import grid_opt.utils.utils_scannet as utils_scannet
from grid_opt.utils.utils_scannet import *
import grid_opt.utils.utils_eval as utils_eval
import grid_opt.local_opt as local_opt
from grid_opt.slam.system import System
from grid_opt.slam.fuser import Fuser
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to config file.', default='./configs/rgbd/scannet.yaml')
parser.add_argument('--default_config', type=str, help='Path to config file.', default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/demo/slam')
parser.add_argument('--scannet_root', type=str, default='data/ScanNet/scans')
parser.add_argument('--scene', type=str, default='0000_00')
parser.add_argument('--mapping_only', action='store_true')



def run_slam(scene: SceneMetadata, 
                    label='incremental',
                    disable_tracking=False, 
                    disable_mapping=False,
                    disable_vis=False):
    # args.config = './configs/rgbd/scannet.yaml' 
    cfg = load_config(args.config, args.default_config)
    log_dir = join(args.save_dir, scene.name, label)
    cond_mkdir(log_dir)
    cfg['dataset']['path'] = scene.path
    cfg['dataset']['intrinsics_file'] = scene.intrinsics_file
    cfg['model']['grid']['bound'] = scene.bound
    cfg['model']['pose']['num_poses'] = scene.num_kfs
    cfg['model']['pose']['optimize'] = True
    cfg['dataset']['anchor_kfs'] = scene.anchor_kfs
    cfg['train']['log_dir'] = log_dir
    grid_atlas = GridAtlas(cfg['model'], device=cfg['device'], dtype=torch.float32) 
    grid_atlas.to(cfg['device'])
    # =====================
    #  Configure datasets 
    # =====================
    # initial n_rays 500000
    dataset_track = utils_scannet.create_scannet_dataset(args.scannet_root, 
                                                         scene.name,
                                                         n_rays=5000,
                                                         n_strat_samples=0,
                                                         n_surf_samples=1,
                                                         voxel_size=0.05,
                                                         frame_downsample=1)
    
    dataset_map = utils_scannet.create_scannet_dataset(args.scannet_root,
                                                       scene.name,
                                                       n_rays=5000,
                                                       n_strat_samples=3,
                                                       n_surf_samples=4,
                                                       voxel_size=0.01,
                                                       frame_downsample=1)
    # =====================
    #  Configure system 
    # =====================
    cfg_s = deepcopy(cfg)
    cfg_s['system']['submap_size'] = scene.num_kfs  # Use a single submap
    cfg_s['system']['log_dir'] = log_dir
    cfg_s['tracking']['disable'] = disable_tracking
    cfg_s['mapping']['disable'] = disable_mapping
    cfg_s['visualizer']['enable'] = not disable_vis
    # =====================
    #  SLAM 
    # =====================
    R0, t0 = dataset_track.true_kf_pose_in_world(0)
    T0 = utils_geometry.pose_matrix(R0, t0)
    system = System(
        model=grid_atlas,
        dataset_track=dataset_track,
        dataset_map=dataset_map,
        cfg=cfg_s,
        R_world_origin=R0, 
        t_world_origin=t0,
        verbose=True
    )
    system.run()

    # =====================
    #  Fusion (debugging) 
    # =====================
    # fuser = Fuser(
    #     model=grid_atlas,
    #     dataset=dataset_map,
    #     cfg=cfg_m
    # )
    # fuser.fuse(
    #     feat_lr=1e-3,
    #     submap_pose_lr=0,
    #     kf_pose_lr=0,
    #     iterations=50
    # )
    info = {'total_compute_time': 0.0}
    model_path = join(log_dir, f'result.pth')
    torch.save(grid_atlas, model_path)
    mesh_path = join(log_dir, f'mesh.ply')
    # save_mesh(grid, grid.bound, mesh_path, resolution=512, transform=T0)
    save_mesh(grid_atlas, grid_atlas.global_bound(), mesh_path, resolution=512)
    return grid_atlas, info


def evaluate_localization_quality(model: BaseNet, scene:SceneMetadata):
    # args.config = './configs/rgbd/scannet.yaml'
    cfg = load_config(args.config, args.default_config)
    cfg['dataset']['path'] = scene.path
    cfg['dataset']['intrinsics_file'] = scene.intrinsics_file
    cfg['dataset']['anchor_kfs'] = scene.anchor_kfs
    # dataset = ScanNet(cfg)
    dataset = utils_scannet.create_scannet_dataset(args.scannet_root, 
                                                   scene.name,
                                                   frame_downsample=1)
    R_true = utils_geometry.identity_rotations(dataset.num_kfs)
    t_true = torch.zeros((dataset.num_kfs, 3, 1))
    R_sol = utils_geometry.identity_rotations(dataset.num_kfs)
    t_sol = torch.zeros((dataset.num_kfs, 3, 1))
    for kf_id in range(dataset.num_kfs):
        Rwk, twk = model.updated_kf_pose_in_world(kf_id)
        Rwk_true, twk_true = dataset.true_kf_pose_in_world(kf_id)
        R_sol[kf_id] = Rwk
        t_sol[kf_id] = twk
        R_true[kf_id] = Rwk_true
        t_true[kf_id] = twk_true
    metrics_t = utils_eval.evo_trajectory_error(R_true, t_true, R_sol, t_sol, align=True, 
                                                pose_relation=evo_metrics.PoseRelation.translation_part).get_all_statistics()
    metrics_R = utils_eval.evo_trajectory_error(R_true, t_true, R_sol, t_sol, align=True,
                                                pose_relation=evo_metrics.PoseRelation.rotation_part).get_all_statistics()
    # save the estimated and ground truth poses
    pose_estimation = {
        'R_est': R_sol,
        't_est': t_sol,
        'R_gt': R_true,
        't_gt': t_true,
    }
    save_file = join(args.save_dir, scene.name, 'pose_estimation.pt')
    print('saving the pose to:', save_file)
    torch.save(pose_estimation, save_file)
    metrics = {
        'rmse_tran (cm)': 100 * metrics_t['rmse'],
        'rmse_deg': utils_geometry.chordal_to_degree(metrics_R['rmse']),
    }
    return metrics

def evaluate_mapping_quality(model: BaseNet, scene:SceneMetadata):
    # note(yulun): this function assumes that the model is already aligned with the ground truth mesh.
    output_mesh_file = join(args.save_dir, scene.name, 'mesh_eval.ply')
    print('Saving the mesh to:', output_mesh_file)
    save_mesh(model, model.bound, output_mesh_file, resolution=512)
    # align the mesh with the ground truth mesh
    aligned_mesh_file = join(args.save_dir, scene.name, 'mesh_eval_aligned.ply')
    mesh_est = o3d.io.read_triangle_mesh(output_mesh_file)
    mesh_gt = o3d.io.read_triangle_mesh(scene.gt_mesh)
    mesh_est_aligned = utils_scannet.align_mesh_to_ref(mesh_est, mesh_gt)
    o3d.io.write_triangle_mesh(aligned_mesh_file, mesh_est_aligned)
    # Compute chamfer distance
    file_pred = aligned_mesh_file
    file_trgt = scene.gt_mesh
    mesh_sample_point = 1000000
    voxel_down_sample_res = 0.02
    verts_pred = utils_eval.sample_points_from_mesh(file_pred, mesh_sample_point=mesh_sample_point, voxel_down_sample_res=voxel_down_sample_res)
    verts_trgt = utils_eval.sample_points_from_mesh(file_trgt, mesh_sample_point=mesh_sample_point, voxel_down_sample_res=voxel_down_sample_res)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(verts_pred)
    # o3d.visualization.draw_geometries([point_cloud])

    # Filter points using GT oriented bounding box
    mesh = o3d.io.read_triangle_mesh(scene.gt_mesh)
    obb = mesh.get_minimal_oriented_bounding_box()
    verts_pred = utils_eval.filter_points_by_oriented_bound(verts_pred, obb)
    verts_trgt = utils_eval.filter_points_by_oriented_bound(verts_trgt, obb)

    # Compute and return metric
    metrics = utils_eval.compute_chamfer_metrics(verts_pred, verts_trgt, threshold=0.05, truncation_acc=0.50, truncation_com=0.50)
    return metrics

def run_on_dataset(scene, methods):
    results = {}
    for method_name, method_func in methods.items():
        model, method_info = method_func(scene)
        metrics = evaluate_mapping_quality(model, scene)
        metrics.update(evaluate_localization_quality(model, scene))
        metrics["time_sec"] = method_info['total_compute_time']
        formatted_metrics = {key: round(value, 2) for key, value in metrics.items()}
        results[method_name] = formatted_metrics
    for method_name, method_result in results.items():
        print(f"Result for method {method_name}:")
        print(json.dumps(method_result, indent=4))
    output_file = join(args.save_dir, scene.name, 'results.json')
    with open(output_file, "w") as f:
        json.dump(results, f, indent = 4)
    print(f"Saved results for {scene.name} to {output_file}.")


def visualize_mesh_for_scene(scene: SceneMetadata, mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)  
    mesh.compute_vertex_normals()
    ref_mesh = o3d.io.read_triangle_mesh(scene.gt_mesh)
    ref_obb = ref_mesh.get_minimal_oriented_bounding_box()
    mesh = mesh.crop(ref_obb)

    def capture_screenshot(vis):
        """
        Capture a screenshot of the current Open3D visualizer window.
        """
        output_file = join(args.save_dir, f"capture.png", )
        vis.capture_screen_image(output_file)
        print(f"Screenshot saved to {output_file}")
        return False  # Continue running the visualizer

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1920, height=1080, visible=True)  # Create a large resolution window
    
    vis.add_geometry(mesh)

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = False
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # Set white background
    render_option.point_size = 5  # Increase point size for better visibility, if any points
    vis.poll_events()
    vis.update_renderer()
    vis.register_key_callback(ord("S"), capture_screenshot)
    vis.run()


if __name__ == "__main__":

    args = parser.parse_args()
    RMAX = 10   # sensing range
    
    scenes_dict = scannet_scenes()
    # Define all methods to evaluate
    visualize = True
    
    incremental_icp_mapping = lambda scene: run_slam(
        scene=scene, 
        label='incremental_icp_mapping',
        disable_tracking=True,
        disable_mapping=False,
        disable_vis=False,
    )
    incremental_slam = lambda scene: run_slam(
        scene=scene, 
        label='incremental_slam',
        disable_tracking=False,
        disable_mapping=False,
        disable_vis=False,
    )
    if args.mapping_only:
        methods = {
        'incremental_icp_mapping': incremental_icp_mapping,
        }
    else:
        methods = {
        'incremental_slam': incremental_slam,
        }

    # Run all methods on different scannet scenes!
    print('Running on scene:', args.scene)
    run_on_dataset(scenes_dict[args.scene], methods=methods)

