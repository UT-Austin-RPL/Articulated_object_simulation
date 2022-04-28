import trimesh
import numpy as np
from s2u.utils.libmesh import check_mesh_contains
from s2u.utils.visual import as_mesh

def sample_iou_points_uni(bounds, num_point, padding=0.1, whole_space=False, size=0.3):
    points = np.random.rand(num_point, 3).astype(np.float32)
    scale = (bounds[1] - bounds[0]).max() * (1 + padding)
    center = (bounds[1] + bounds[0]) / 2
    if whole_space:
        points *= size + 2 * padding
        points -= padding
    else:
        points = (points - 0.5)* scale + center
    # occ = check_mesh_contains(mesh, points)
    return points.astype(np.float16)

def sample_iou_points_surf(mesh, num_point, var=0.01):
    surf_points = mesh.sample(num_point)
    variation = np.random.randn(num_point, 3) * var
    points = surf_points + variation
    # occ = check_mesh_contains(mesh, points)
    return points.astype(np.float16)

def sample_iou_points_uni_surf(
    mesh_list,
    bounds,
    num_point,
    padding=0.1,
    var=0.01
):
    num_point_uniform = num_point // 4
    num_point_surface = num_point - num_point_uniform
    points_uniform = np.random.rand(num_point_uniform, 3).astype(np.float32) - 0.5
    scale = (bounds[1] - bounds[0]).max() * (1 + padding)
    center = (bounds[1] + bounds[0]) / 2
    points_uniform = points_uniform * scale + center
    num_point_surface_per_mesh = num_point_surface // (len(mesh_list) * 2)
    points_surface = []
    for mesh in mesh_list:
        for var_local in [var * scale, var * scale * 0.1]:
            points_surface_mesh = mesh.sample(num_point_surface_per_mesh)
            # add variation
            variation = np.random.randn(num_point_surface_per_mesh, 3) * var_local
            points_surface_mesh = points_surface_mesh + variation
            points_surface.append(points_surface_mesh)

    points_surface.append(points_uniform)

    points = np.concatenate(points_surface, axis=0)
    return points.astype(np.float16)

def sample_iou_points_occ(mesh_list, bounds, num_point, method, padding=0.1, var=0.01):
    if method == 'uniform':
        points = sample_iou_points_uni(bounds, num_point, padding)
    elif method == 'surface':
        full_mesh = as_mesh(trimesh.Scene(mesh_list))
        points = sample_iou_points_surf(full_mesh, num_point, var)
    elif method == 'mix':
        points = sample_iou_points_uni_surf(mesh_list, bounds, num_point, padding, var)
    occ_list = []
    for mesh in mesh_list:
        occ = check_mesh_contains(mesh, points)
        occ_list.append(occ)
    return points.astype(np.float16), occ_list
