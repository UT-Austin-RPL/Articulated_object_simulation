import numpy as np
import trimesh
from sklearn.neighbors import KDTree
from scipy.spatial import KDTree as KDTree_scipy
from s2u.utils.visual import as_mesh

def sample_point_cloud(pc, num_point):
    num_point_all = pc.shape[0]
    idxs = np.random.choice(np.arange(num_point_all), size=(num_point,), replace=num_point > num_point_all)
    return pc[idxs]

def collect_density_per_object(anchors, points, values, gridnum=35):
    pos_neighbors = np.zeros((anchors.shape[0], 1))
    neg_neighbors = np.zeros((anchors.shape[0], 1))

    step = 1.0 / gridnum
    tree = KDTree(points)
    inds = tree.query_radius(anchors, r=step / 2.0)
    for p_id in range(anchors.shape[0]):
        nlist = inds[p_id]
        if (len(nlist) > 0):
            vs = values[nlist]
            posnum = np.sum(vs < 0)
            negnum = np.sum(vs > 0)
        else:
            posnum = 0
            negnum = 0
        pos_neighbors[p_id, 0] = posnum
        neg_neighbors[p_id, 0] = negnum

    return pos_neighbors, neg_neighbors


def estimate_density(points, values):
    # compute the inner/outer density
    # occ -0.5 to {-0.5, 0.5}
    bound_max, bound_min = np.max(points, 0), np.min(points, 0)
    center = (bound_max + bound_min) / 2
    scale = (bound_max - bound_min).max()
    normed_points = (points - center) / scale
    
    if values.min() < 0:
        tmp_val = values
    else:
        tmp_val = values - 0.5
    pos_neighbors, neg_neighbors = collect_density_per_object(normed_points,
                                                              normed_points,
                                                              tmp_val,
                                                              gridnum=20)
    densities = np.zeros((points.shape[0], 1))
    for i in range(points.shape[0]):
        v = values[i]
        if (v < 0):
            densities[i] = 1 / (pos_neighbors[i] + 0.01)
        else:
            densities[i] = 1 / (neg_neighbors[i] + 0.01)

    return densities

def norm_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    bounds = mesh.bounds
    center = bounds.mean(0)
    scale = (bounds[1] - bounds[0]).max()
    transform_mat = np.eye(4)
    transform_mat[:3, 3] = -center
    transform_mat[:3] /= scale
    mesh.apply_transform(transform_mat)
    return mesh

def get_labeled_surface_pc(mesh_pose_dict, mobile_links, num_point=2048):
    mesh_dict = {'s': trimesh.Scene(), 'm': trimesh.Scene()}
    for k, v in mesh_pose_dict.items():
        for mesh_path, scale, pose in v:
            if mesh_path.startswith('#'): # primitive
                mesh = trimesh.creation.box(extents=scale, transform=pose)
            else:
                mesh = trimesh.load(mesh_path)
                mesh.apply_scale(scale)
                mesh.apply_transform(pose)
            if int(k.split('_')[1]) in mobile_links:
                mesh_dict['m'].add_geometry(mesh)
            else:
                mesh_dict['s'].add_geometry(mesh)
    sample_points = []
    labels = []
    sample_points.append(as_mesh(mesh_dict['s']).sample(num_point))
    labels.append(np.zeros(num_point))
    sample_points.append(as_mesh(mesh_dict['m']).sample(num_point))
    labels.append(np.ones(num_point))
    sample_points = np.concatenate(sample_points, 0)
    labels = np.concatenate(labels, 0)
    return sample_points, labels

def segment_pc(pc, mesh_pose_dict, mobile_links):
    surface_pc, label = get_labeled_surface_pc(mesh_pose_dict, mobile_links)
    tree = KDTree_scipy(surface_pc)
    _, nearest_neighbor_idxs = tree.query(pc)
    seg_label = label[nearest_neighbor_idxs].astype(bool)
    return seg_label