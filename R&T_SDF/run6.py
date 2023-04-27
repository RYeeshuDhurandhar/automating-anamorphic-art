from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
import numpy as np
from scipy.spatial.transform import Rotation


mesh = trimesh.load('/home/vrajesh/Documents/DeepSDF/mesh_to_sdf-master/example/toy.obj')


points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)


colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1


bbox = trimesh.creation.box(extents=[2, 2, 2])
bbox_mesh = pyrender.Mesh.from_trimesh(bbox, smooth=False, wireframe=True)


cloud = pyrender.Mesh.from_points(points, colors=colors)


scene = pyrender.Scene()

cloud_node = scene.add(cloud)


translation = np.identity(4)
translation[:3, 3] = [0, 2, 0]  
rotation_matrix = np.identity(4)
rotation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True)
rotation2 = rotation.as_matrix()
rotation_matrix[:3, :3] = rotation2
# rotation_matrix = np.vstack((rotation_matrix, [0, 0, 0, 1]))
cloud_node.matrix = np.matmul(translation, np.matmul(rotation_matrix, cloud_node.matrix))
new_points = np.matmul(rotation_matrix[:3, :3], points.T).T + translation[:3, 3]

bbox_center = np.array([0, 0, 0])
bbox_min = bbox_center - np.array([1.0, 1.0, 1.0])
bbox_max = bbox_center + np.array([1.0, 1.0, 1.0])

valid_points = np.all((new_points >= bbox_min) & (new_points <= bbox_max), axis=1)
valid_points_indices = np.where(valid_points)[0]
valid_points_cloud = pyrender.Mesh.from_points(new_points[valid_points_indices], colors=colors[valid_points_indices])

scene2 = pyrender.Scene()
bbox_node = scene2.add(bbox_mesh)

filtered_cloud_node = scene2.add(valid_points_cloud)


viewer = pyrender.Viewer(scene2, use_raymond_lighting=True, point_size=2)
