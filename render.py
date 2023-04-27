# import torch
# import torch.nn.functional as F
# import matplotlib as plt

# def sphere_sdf(x: torch.Tensor, r: float = 1.0) -> float:
#     """
#     Returns the signed distance to a sphere centered at the origin with the given radius.
#     """
#     return torch.norm(x, dim=-1) - r


# def look_at_rotation(position: torch.Tensor,
#                      target: torch.Tensor,
#                      up: torch.Tensor) -> torch.Tensor:
#     """
#     Returns a rotation matrix that orients an object to face the given target direction, with the given up direction.
#     """
#     forward = F.normalize(target - position, dim=-1)
#     right = torch.cross(forward, up, dim=-1)
#     true_up = torch.cross(right, forward, dim=-1)
#     rotation = torch.stack([right, true_up, -forward], dim=-1)
#     return rotation


# def perspective_right_handed(fov_y: torch.Tensor,
#                              aspect_ratio: torch.Tensor,
#                              near: torch.Tensor,
#                              far: torch.Tensor) -> torch.Tensor:
#     """
#     Returns a perspective projection matrix for a right-handed coordinate system.
#     """
#     y_scale = 1.0 / torch.tan(0.5 * fov_y)
#     x_scale = y_scale / aspect_ratio
#     depth_scale = far / (far - near)
#     depth_bias = -near * depth_scale
#     return torch.tensor([
#         [x_scale, 0.0, 0.0, 0.0],
#         [0.0, y_scale, 0.0, 0.0],
#         [0.0, 0.0, depth_scale, -1.0],
#         [0.0, 0.0, depth_bias * depth_scale, 0.0]
#     ], dtype=torch.float32, device=fov_y.device)


# def ray_march(origins: torch.Tensor,
#               directions: torch.Tensor,
#               scene_sdf,
#               max_iters: int = 1000,
#               epsilon: float = 1e-6) -> torch.Tensor:
#     """
#     Performs ray marching to find the intersection points between rays and a Signed Distance Function (SDF).

#     Args:
#         origins: A tensor of shape (num_rays, 3) representing the origins of the rays.
#         directions: A tensor of shape (num_rays, 3) representing the directions of the rays.
#         scene_sdf: A callable that computes the Signed Distance Function (SDF) for the scene.
#         max_iters: The maximum number of iterations to perform.
#         epsilon: The tolerance for the distance to the surface before considering it an intersection.

#     Returns:
#         A tensor of shape (num_rays, 3) containing the intersection points on each ray, or -1 if no intersection was found.
#     """
#     dists = torch.zeros_like(origins[..., :1])
#     for i in range(max_iters):
#         dists += scene_sdf(origins + directions * dists) * directions.norm(dim=-1, keepdim=True)
#         mask = (dists > epsilon) & (dists.norm(dim=-1) < 100.0)
#         if not mask.any():
#             break
#     return torch.where(mask, origins + directions * dists, -torch.ones_like(origins))

# def render_sphere_sdf(radius: float = 1.0,
#                       num_pixels: int = 256,
#                       fov_degrees: float = 60.0,
#                       device: str = 'cpu') -> torch.Tensor:
#     """
#     Renders a sphere represented by a Signed Distance Function (SDF) using ray marching.

#     Args:
#         radius: The radius of the sphere.
#         num_pixels: The number of pixels in the output image (assumes a square image).
#         fov_degrees: The field of view in degrees for the camera.
#         device: The device on which to perform computations (e.g., 'cpu' or 'cpu:0').

#     Returns:
#         A tensor of shape (3, num_pixels, num_pixels) containing the rendered image.
#     """
#     # Set up the camera
#     fov = torch.tensor([fov_degrees], dtype=torch.float32, device=device)
#     aspect_ratio = torch.tensor([1.0], dtype=torch.float32, device=device)
#     camera_position = torch.tensor([0.0, 0.0, -2.0 * radius], dtype=torch.float32, device=device)
#     camera_direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

#     # Compute the pixel coordinates in the image plane
#     x = torch.linspace(-1.0, 1.0, num_pixels, dtype=torch.float32, device=device)
#     y = torch.linspace(-1.0, 1.0, num_pixels, dtype=torch.float32, device=device)
#     px, py = torch.meshgrid(x, y)
#     pixel_coords = torch.stack([px.reshape(-1), py.reshape(-1), torch.ones_like(px.reshape(-1))], dim=1)

#     # Compute view matrix and projection matrix
#     view_matrix = look_at_rotation(camera_position, camera_position + camera_direction, up=torch.tensor([0.0, 1.0, 0.0], device=device)).transpose(0, 1)
#     projection_matrix = perspective_right_handed(fov, aspect_ratio, 0.1 * radius, 10.0 * radius)
#     view_matrix_4x4 = torch.cat([view_matrix, torch.tensor([[0.0, 0.0, 0.0]], device=device)], dim=0)
#     view_matrix_4x4 = torch.cat([view_matrix_4x4, torch.tensor([[0.0], [0.0], [0.0], [1.0]], device=device)], dim=1)



#     # Compute ray directions
#     view_projection_matrix = torch.inverse(projection_matrix) @ view_matrix_4x4
#     view_projection_matrix = view_projection_matrix[:3, :3]
#     ray_directions = F.normalize(torch.matmul(pixel_coords, view_projection_matrix.transpose(0, 1))[:, :3], dim=1)

#     # Render the scene
#     scene_sdf = lambda x: sphere_sdf(x, radius)
#     intersection_points = ray_march(camera_position.unsqueeze(0).expand(num_pixels**2, -1),
#                                     ray_directions,
#                                     scene_sdf)
#     intersection_points = intersection_points.reshape(num_pixels, num_pixels, 3).permute(2, 0, 1)

#     # Compute the distance from each pixel to the intersection point on its ray
#     pixel_distances = torch.norm(pixel_coords[:, :2] - torch.tensor([0.5, 0.5], dtype=torch.float32, device=device), dim=-1).reshape(num_pixels, num_pixels)
#     intersection_distances = torch.norm(intersection_points - camera_position.unsqueeze(-1), dim=0)

#     # Compute the rendered image
#     image = 1.0 - torch.clamp(intersection_distances / radius, min=0.0, max=1.0)
#     image[pixel_distances > 1.0] = 0.0
#     image = image.unsqueeze(0).repeat(3, 1, 1)

#     return image
# image = render_sphere_sdf(radius=1.0, num_pixels=256, fov_degrees=60.0, device='cpu')
# plt.imshow(image.permute(1, 2, 0).cpu().numpy())
# plt.show()

import torch
import pytorch3d
# from pytorch3d.renderer import SDFRenderer
# from pytorch3d.structures import SDF
import torch

def sdf_renderer(sdf_fn, image_size=256, num_samples=64, camera_position=None):
    # Create a grid of image_size x image_size x 3 pixel coordinates
    x = torch.linspace(-1.0, 1.0, image_size).view(1, -1, 1).repeat(image_size, 1, 1)
    y = torch.linspace(-1.0, 1.0, image_size).view(-1, 1, 1).repeat(1, image_size, 1)
    z = torch.zeros_like(x)
    coords = torch.stack((x, y, z), dim=-1)

    # Evaluate the SDF at each pixel coordinate
    sdf_vals = sdf_fn(coords.view(-1, 3)).view(image_size, image_size)

    # Create a mask of pixels that lie inside the SDF surface
    mask = (sdf_vals <= 0.0).float()

    # Sample num_samples random points inside the SDF surface
    if mask.sum() > 0.0:
        samples = torch.randn(num_samples, 3)
        samples /= torch.norm(samples, dim=-1, keepdim=True)
        samples *= sdf_fn(samples).abs().max()
        samples = samples[(sdf_fn(samples) <= 0.0).nonzero(as_tuple=False).squeeze()]
    else:
        samples = None

    # Compute the distance from each pixel to the SDF surface
    sdf_dists = torch.abs(sdf_vals)

    # Compute the color of each pixel based on the distance to the SDF surface
    colors = torch.cat((mask.unsqueeze(-1), mask.unsqueeze(-1), mask.unsqueeze(-1)), dim=-1)
    colors = colors.clamp(0.0, 1.0)

    # Apply shading to the colors using the camera position
    # if camera_position is not None and samples is not None:
    #     normals = torch.zeros_like(samples)
    #     eps = 1e-6
    #     for i in range(num_samples):
    #         x1 = samples[i] + torch.tensor([eps, 0.0, 0.0])
    #         x2 = samples[i] + torch.tensor([0.0, eps, 0.0])
    #         x3 = samples[i] + torch.tensor([0.0, 0.0, eps])
    #         n = torch.stack([
    #             sdf_fn(x1) - sdf_fn(samples[i]),
    #             sdf_fn(x2) - sdf_fn(samples[i]),
    #             sdf_fn(x3) - sdf_fn(samples[i])
    #         ])
    #         n /= n.norm(dim=-1, keepdim=True)
    #         normals[i] = n
    #     camera_dir = camera_position - samples
    #     camera_dir /= camera_dir.norm(dim=-1, keepdim=True)
    #     light_dir = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, -1).repeat(num_samples, 1, 1)
    #     diffuse = (normals * light_dir).sum(dim=-1).clamp(0.0, 1.0)
    #     # print(diffuse.shape)
    #     # print(mask)
    #     # diffuse *= mask.unsqueeze(-1).repeat(1, 1, 3)
    #     colors = colors * diffuse.unsqueeze(-2)

    # Convert the colors to an image tensor
    image = colors.permute(2, 0, 1)

    return image


import torch
import pytorch3d


# Define the SDF function for a sphere
def sphere_sdf(xyz, radius=1.0):
    return torch.norm(xyz, dim=-1) - radius

# Create a SDFRenderer object
# renderer = sdf_renderer(sphere_sdf, image_size=256, num_samples=64, camera_position=None)

# Create a tensor of points to evaluate the SDF function at
x, y, z = torch.meshgrid(torch.linspace(-1, 1, 64),
                          torch.linspace(-1, 1, 64),
                          torch.linspace(-1, 1, 64))
points = torch.stack([x, y, z], dim=-1).view(-1, 3)

# Evaluate the SDF function at the points
sdf_values = sphere_sdf(points)

# Create an SDF object using the values
# sdf = SDFVoxelGrid(sdf_values.view(1, 64, 64, 64), voxel_size=2./64, sdf_trunc=3*2./64)

# Define the camera position and orientation
R = torch.eye(3)
T = torch.tensor([0.0, 0.0, 3.0])
camera_position = -R.T @ T

# Render the sphere using the SDFRenderer
image = sdf_renderer(sphere_sdf, image_size=256, num_samples=64, camera_position=camera_position)
image = image.permute(1, 2, 0).cpu().numpy()
# Show the rendered image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()

