import kaolin as kal
import numpy as np
import torch
import trimesh
import pyrender

mesh = kal.rep.TriangleMesh.from_obj('Spider_Monkey.obj')

mesh.cuda()


sdf = kal.conversions.trianglemesh_to_sdf(mesh)

points = mesh.sample(100)[0]

d = sdf(points)

colors = np.zeros(points.shape)

d = d.detach().cpu()

points = points.detach().cpu()

colors[d < 0, 2] = 1

colors[d > 0, 0] = 1

cloud = pyrender.Mesh.from_points(points, colors=colors)

scene = pyrender.Scene()

scene.add(cloud)

viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
