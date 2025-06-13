import torch
import cuda_gridsample as cu

# Create a 3D input volume of shape (1, 1, D, H, W)
D, H, W = 4, 4, 4
# input_volume = torch.arange(D * H * W, dtype=torch.float32).view(1, 1, D, H, W)
input_volume = torch.randn(1, 1, D, H, W)

# Define a 3D grid for sampling of shape (1, D_out, H_out, W_out, 3)
D_out, H_out, W_out = 2, 2, 2
grid = torch.stack(torch.meshgrid(
    torch.linspace(-1, 1, D_out),
    torch.linspace(-1, 1, H_out),
    torch.linspace(-1, 1, W_out),
    indexing="ij"
), dim=-1).unsqueeze(0)  # Shape: (1, D_out, H_out, W_out, 3)

# Perform 3D grid sampling
output_volume = torch.nn.functional.grid_sample(input_volume, grid, mode='bilinear', padding_mode='border', align_corners=False)
output_volume2 = cu.grid_sample_3d(input_volume, grid, padding_mode='border', align_corners=False)

# Print the results
print(torch.linalg.norm(output_volume - output_volume2))
print(output_volume2.squeeze())