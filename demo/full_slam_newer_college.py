import argparse
from os.path import join
from grid_opt.configs import *
from grid_opt.datasets.sdf_3d_lidar import PosedSdf3DLidar
from grid_opt.utils.utils_sdf import *
from grid_opt.slam.system import System
import logging
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--default_config', type=str, help='Path to config file.', default='./configs/base.yaml')
parser.add_argument('--save_dir', type=str, default='./results/demo/slam/ncd_quad')
parser.add_argument('--run_name', type=str, default='test')
parser.add_argument('--pose_init', type=str, default='reg_icp')  # reg_icp OR kiss_icp OR gt
parser.add_argument('--config', type=str, help='Path to config file.', default='./configs/lidar/ncd_quad.yaml')


def create_ncd_dataset(
        cfg, 
        voxel_size=0.03, 
        near_surf_std=0.1, 
        n_near=4, 
        n_free=2, 
        n_behind=1,
        frame_samples=2**12,
        frame_batchsize=2**10
    ):
    dataset = PosedSdf3DLidar(
        lidar_folder=cfg['dataset']['path'],
        pose_file_gt=cfg['dataset']['pose_gt'],
        pose_file_init=cfg['dataset']['pose_init'],
        num_frames=cfg['dataset']['num_frames'],
        trunc_dist=cfg['dataset']['trunc_dist'],
        frame_samples=frame_samples,
        frame_batchsize=frame_batchsize,
        voxel_size=voxel_size,
        near_surface_std=near_surf_std,
        near_surface_n=n_near,
        free_space_n=n_free,
        behind_surface_n=n_behind,
        min_dist_ratio=0.50,
        min_z=-10.0,
        max_z=60.0,
        min_range=1.5, 
        max_range=60.0,
        adaptive_range=False
    )
    return dataset


def create_configs(cfg, rmax):
    # Tracking settings
    cfg['tracking']['verbose'] = False
    # Mapping settings
    cfg['mapping']['learning_rate'] = 1e-3
    cfg['mapping']['sigmoid_scale'] = 0.001 * rmax
    cfg['mapping']['verbose'] = False
    # Related to mapping pool
    cfg['mapping']['max_replay_frames'] = 5
    cfg['mapping']['max_replay_freq'] = 10
    # System setting
    cfg['system']['log_dir'] = join(args.save_dir, args.run_name)
    cfg['system']['submap_size'] = 2000
    cfg['system']['submap_local_bound'] = [[-60, 60], [-60, 60], [-5, 15]]
    # Pose initialization setting
    if args.pose_init == 'reg_icp':
        cfg['dataset']['pose_init'] = join(cfg['dataset']['root'], 'poses_reg_icp.txt')
    elif args.pose_init == 'kiss_icp':
        cfg['dataset']['pose_init'] = join(cfg['dataset']['root'], 'poses_kiss_icp.txt')
    elif args.pose_init == 'gt':
        cfg['dataset']['pose_init'] = join(cfg['dataset']['root'], 'poses_gt.txt')
    else:
        raise ValueError(f"Unknown pose initialization method: {args.pose_init}")
    return cfg


def run_slam(grid_atlas, cfg):
    log_dir = join(args.save_dir, args.run_name)
    cond_mkdir(log_dir)
    dataset_track = create_ncd_dataset(
        cfg, 
        voxel_size=0.6, 
        n_near=0, n_free=0, n_behind=0, 
        frame_samples=2**20, frame_batchsize=2**14
    )
    dataset_map = create_ncd_dataset(
        cfg, 
        voxel_size=0.08, 
        near_surf_std=0.25,
        n_near=4, n_free=2, n_behind=1, 
        frame_samples=2**12
    )
    system = System(
        model=grid_atlas,
        dataset_track=dataset_track,
        dataset_map=dataset_map,
        cfg=cfg,
        verbose=True
    )
    system.run()
    model_path = join(log_dir, f'odometry.pth')
    torch.save(grid_atlas, model_path)
    return grid_atlas


def main():
    cfg = load_config(args.config, args.default_config)
    cfg = create_configs(cfg, rmax=60.0)
    grid_atlas = GridAtlas(cfg['model'], device=cfg['device'], dtype=torch.float32) 
    grid_atlas.to(cfg['device'])
    run_slam(grid_atlas, cfg)
    # Save final higher resolution mesh
    mesh_path = join(args.save_dir, args.run_name, 'final_mesh.ply')
    save_mesh(grid_atlas, grid_atlas.global_bound(), mesh_path, resolution=512)


if __name__ == "__main__":
    args = parser.parse_args()
    main()

