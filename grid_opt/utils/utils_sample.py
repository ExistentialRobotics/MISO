# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
# from isdf.geometry import transform

def ray_dirs_C(B, H, W, fx, fy, cx, cy, device, depth_type='z'):
    c, r = torch.meshgrid(torch.arange(W, device=device),
                          torch.arange(H, device=device))
    c, r = c.t().float(), r.t().float()
    size = [B, H, W]

    C = torch.empty(size, device=device)
    R = torch.empty(size, device=device)
    C[:, :, :] = c[None, :, :]
    R[:, :, :] = r[None, :, :]

    z = torch.ones(size, device=device)
    x = (C - cx) / fx
    y = (R - cy) / fy

    dirs = torch.stack((x, y, z), dim=3)
    if depth_type == 'euclidean':
        norm = torch.norm(dirs, dim=3)
        dirs = dirs * (1. / norm)[:, :, :, None]

    return dirs


def origin_dirs_W(T_WC, dirs_C):
    R_WC = T_WC[:, :3, :3]
    dirs_W = (R_WC * dirs_C[..., None, :]).sum(dim=-1)
    origins = T_WC[:, :3, -1]

    return origins, dirs_W


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


def sample_pixels(
    n_rays, n_frames, h, w, device
):
    total_rays = n_rays * n_frames
    indices_h = torch.randint(0, h, (total_rays,), device=device)
    indices_w = torch.randint(0, w, (total_rays,), device=device)

    indices_b = torch.arange(n_frames, device=device)
    indices_b = indices_b.repeat_interleave(n_rays)

    return indices_b, indices_h, indices_w


def get_batch_data(
    depth_batch,
    T_WC_batch,
    dirs_C,
    indices_b,
    indices_h,
    indices_w,
    norm_batch=None,
    get_masks=False,
):
    """
    Get depth, ray direction and pose for the sampled pixels.
    Only render where depth is valid.
    """
    depth_sample = depth_batch[indices_b, indices_h, indices_w].view(-1)
    mask_valid_depth = depth_sample != 0

    norm_sample = None
    if norm_batch is not None:
        norm_sample = norm_batch[indices_b,
                                 indices_h,
                                 indices_w, :].view(-1, 3)
        mask_invalid_norm = torch.isnan(norm_sample[..., 0])
        mask_valid_depth = torch.logical_and(
            mask_valid_depth, ~mask_invalid_norm)
        norm_sample = norm_sample[mask_valid_depth]

    depth_sample = depth_sample[mask_valid_depth]

    indices_b = indices_b[mask_valid_depth]
    indices_h = indices_h[mask_valid_depth]
    indices_w = indices_w[mask_valid_depth]

    T_WC_sample = T_WC_batch[indices_b]
    dirs_C_sample = dirs_C[0, indices_h, indices_w, :].view(-1, 3)

    masks = None
    if get_masks:
        masks = torch.zeros(depth_batch.shape, device=depth_batch.device)
        masks[indices_b, indices_h, indices_w] = 1

    return (
        dirs_C_sample,
        depth_sample,
        norm_sample,
        T_WC_sample,
        masks,
        indices_b,
        indices_h,
        indices_w
    )


def stratified_sample(
    min_depth,
    max_depth,
    n_rays,
    device,
    n_stratified_samples,
    bin_length=None,
):
    """
    Random samples between min and max depth
    One sample from within each bin.

    If n_stratified_samples is passed then use fixed number of bins,
    else if bin_length is passed use fixed bin size.
    """
    if n_stratified_samples is not None:  # fixed number of bins
        n_bins = n_stratified_samples
        if isinstance(max_depth, torch.Tensor):
            sample_range = (max_depth - min_depth)[:, None]
            bin_limits = torch.linspace(
                0, 1, n_bins + 1,
                device=device)[None, :]
            bin_limits = bin_limits.repeat(n_rays, 1) * sample_range
            if isinstance(min_depth, torch.Tensor):
                bin_limits = bin_limits + min_depth[:, None]
            else:
                bin_limits = bin_limits + min_depth
            bin_length = sample_range / (n_bins)
        else:
            bin_limits = torch.linspace(
                min_depth,
                max_depth,
                n_bins + 1,
                device=device,
            )[None, :]
            bin_length = (max_depth - min_depth) / (n_bins)

    elif bin_length is not None:  # fixed size of bins
        bin_limits = torch.arange(
            min_depth,
            max_depth,
            bin_length,
            device=device,
        )[None, :]
        n_bins = bin_limits.size(1) - 1

    increments = torch.rand(n_rays, n_bins, device=device) * bin_length
    # increments = 0.5 * torch.ones(n_rays, n_bins, device=device) * bin_length
    lower_limits = bin_limits[..., :-1]
    z_vals = lower_limits + increments

    return z_vals


def sample_along_rays(
    T_WC,
    min_depth,
    max_depth,
    n_stratified_samples,
    n_surf_samples,
    dirs_C,
    gt_depth=None,
    grad=False,
):
    with torch.set_grad_enabled(grad):
        # rays in world coordinate
        origins, dirs_W = origin_dirs_W(T_WC, dirs_C)

        origins = origins.view(-1, 3)
        dirs_W = dirs_W.view(-1, 3)
        n_rays = dirs_W.shape[0]

        # stratified sampling along rays # [total_n_rays, n_stratified_samples]
        z_vals = stratified_sample(
            min_depth, max_depth,
            n_rays, T_WC.device,
            n_stratified_samples,
            bin_length=None,
        )

        # if gt_depth is given, first sample at surface then around surface
        if gt_depth is not None and n_surf_samples > 0:
            surface_z_vals = gt_depth
            if n_surf_samples == 1:
                # Only sample at the surface
                z_vals = torch.cat(
                    (surface_z_vals[:, None], z_vals), dim=1)
            else:
                # Sample additional points around the surface via Gaussian noise
                offsets = torch.normal(
                    torch.zeros(gt_depth.shape[0], n_surf_samples - 1), 0.1
                ).to(z_vals.device)
                near_surf_z_vals = gt_depth[:, None] + offsets
                if not isinstance(min_depth, torch.Tensor):
                    min_depth = torch.full(near_surf_z_vals.shape, min_depth).to(
                        z_vals.device)[..., 0]
                near_surf_z_vals = torch.clamp(
                    near_surf_z_vals,
                    min_depth[:, None],
                    max_depth[:, None],
                )
                z_vals = torch.cat(
                    (surface_z_vals[:, None], near_surf_z_vals, z_vals), dim=1)

        # point cloud of 3d sample locations
        pc = origins[:, None, :] + (dirs_W[:, None, :] * z_vals[:, :, None])

    return pc, z_vals
