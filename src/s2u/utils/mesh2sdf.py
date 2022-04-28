import mesh_to_sdf
import numpy as np
import trimesh
from mesh_to_sdf import get_surface_point_cloud, mesh_to_sdf
from mesh_to_sdf.utils import sample_uniform_points_in_unit_sphere

def scale_to_unit_sphere(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    center = mesh.bounding_box.centroid
    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    scale = np.max(distances)
    vertices /= scale
    

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces), center, scale

def sample_sdf_near_surface(mesh,
                            number_of_points=500000,
                            surface_point_method='scan',
                            sign_method='normal',
                            scan_count=100,
                            scan_resolution=400,
                            sample_point_count=10000000,
                            normal_sample_count=11,
                            min_size=0,
                            return_gradients=False,
                            uniform=False):
    mesh, center, scale = scale_to_unit_sphere(mesh)

    if surface_point_method == 'sample' and sign_method == 'depth':
        print(
            "Incompatible methods for sampling points and determining sign, using sign_method='normal' instead."
        )
        sign_method = 'normal'
    if uniform:
        point = sample_uniform_points_in_unit_sphere(number_of_points)
        sdf = mesh_to_sdf(mesh, point)
    else:
        surface_point_cloud = get_surface_point_cloud(
            mesh,
            surface_point_method,
            1,
            scan_count,
            scan_resolution,
            sample_point_count,
            calculate_normals=sign_method == 'normal' or return_gradients)

        point, sdf = surface_point_cloud.sample_sdf_near_surface(
            number_of_points, surface_point_method == 'scan', sign_method,
            normal_sample_count, min_size, return_gradients)
    point = point * scale + center
    return point, sdf