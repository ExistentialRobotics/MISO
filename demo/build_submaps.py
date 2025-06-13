import argparse
from os.path import join
import numpy as np
from grid_opt.configs import *
from grid_opt.utils.utils_sdf import *
from grid_opt.slam.mapper import Mapper
from grid_opt.slam.system import System
from grid_opt.datasets.submap_dataset import SubmapDataset
import grid_opt.utils.utils_sdf as utils_sdf
import grid_opt.utils.utils_scannet as utils_scannet
import open3d as o3d
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


def submap_mapping(cfg, grid_atlas:GridAtlas, dataset:SubmapDataset, submap_id):
    cfg['mapping']['verbose'] = True
    cfg['mapping']['disable'] = False
    submap_size = cfg['system']['submap_size']
    frame_start = submap_size * submap_id  
    frame_end = min(submap_size * (submap_id + 1), dataset.num_kfs)
    mapper = Mapper(
        model=grid_atlas.get_submap(submap_id),
        dataset=dataset,
        cfg=cfg
    )
    mapper.mapping(
        mapping_kfs=range(frame_start, frame_end),
        iterations=300,
        level_iterations=50
    )

def save_submap(grid_atlas:GridAtlas, submap_id:int, save_dir=None, visualize=True, postfix=''):
    submap = grid_atlas.get_submap(submap_id)
    R, t = grid_atlas.updated_submap_pose(submap_id)
    T = utils_geometry.pose_matrix(R, t)
    mesh_path = None
    if save_dir is not None:
        mesh_path = join(save_dir, f'submap_{submap_id}.ply')
    mesh = utils_sdf.save_mesh(submap, submap.bound, transform=T, save_path=mesh_path)
    if visualize:
        o3d.visualization.draw_geometries([mesh], window_name=f"Submap {submap_id} {postfix}")

def visualize_submap_split(args, grid_atlas:GridAtlas):
    submap_obbs = []
    for i in range(grid_atlas.num_submaps):
        obb = grid_atlas.submap_obb_in_world(i)
        submap_obbs.append(obb)
    
    gt_mesh_path = join(args.scannet_root, f"scene{args.scene}/scene{args.scene}_vh_clean_2.ply")
    # gt_mesh_tri = trimesh.load(gt_mesh_path)
    # gt_mesh = o3d.geometry.TriangleMesh()
    # gt_mesh.vertices = o3d.utility.Vector3dVector(gt_mesh_tri.vertices)
    # gt_mesh.triangles = o3d.utility.Vector3iVector(gt_mesh_tri.faces)
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    vertices = np.asarray(gt_mesh.vertices)
    # print('x range:', np.min(vertices[:, 0]), np.max(vertices[:, 0]))
    # print('y range:', np.min(vertices[:, 1]), np.max(vertices[:, 1]))
    # print('z range:', np.min(vertices[:, 2]), np.max(vertices[:, 2]))
    o3d.visualization.draw_geometries(
        submap_obbs + [gt_mesh],
        window_name=f"Submaps and GT Mesh for {args.scene}"
    )

def main_scannet():
    np.random.seed(55)
    torch.manual_seed(55)
    args = parser.parse_args()
    model_path = join(args.save_dir, 'grid_atlas.pth')
    cfg, grid_atlas, dataset = initialize_scannet(args)
    
    # Build the submaps from the dataset
    for i in range(grid_atlas.num_submaps):
        submap_mapping(cfg, grid_atlas, dataset, i)
        # visualize the coarse level submap
        grid_atlas.ignore_level(1)  # Ignore level 1 for coarse-level visualization
        save_submap(grid_atlas, i, save_dir=join(args.save_dir, 'submaps'), visualize=True, postfix='Coarse Level')
        grid_atlas.include_level(1)  # Unignore level 1 for fine-level visualization
        save_submap(grid_atlas, i, save_dir=join(args.save_dir, 'submaps'), visualize=True, postfix='Fine Level')
        
    torch.save(grid_atlas, model_path)
    

if __name__ == "__main__":
    main_scannet()
