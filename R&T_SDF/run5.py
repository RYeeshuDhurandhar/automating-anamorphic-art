from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
import numpy as np


mesh = trimesh.load('/home/vrajesh/Documents/DeepSDF/mesh_to_sdf-master/example/toy.obj')

points, sdf = sample_sdf_near_surface(mesh, number_of_points=250000)

colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1

bbox = trimesh.creation.box(extents=[2, 2, 2])
bbox_mesh = pyrender.Mesh.from_trimesh(bbox, smooth=False, wireframe=True)

cloud = pyrender.Mesh.from_points(points, colors=colors)

bbox_center = np.array([0, 0, 0])
bbox_extents = np.array([1.0, 1.0, 1.0])
valid_points = np.all(np.abs(points - bbox_center) < bbox_extents, axis=1)
points = points[valid_points]
colors = colors[valid_points]

cloud = pyrender.Mesh.from_points(points, colors=colors)

scene = pyrender.Scene()
bbox_node = scene.add(bbox_mesh)
cloud_node = scene.add(cloud)

translation = np.identity(4)
translation[:3, 3] = [0.5, 0, 0]  
cloud_node.matrix = np.matmul(translation, cloud_node.matrix)

viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

