import os
import glob
import trimesh
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import multiprocessing as mp

from tqdm import tqdm
import numpy as np

from s2u.simulation import ArticulatedObjectManipulationSim
from s2u.utils.axis2transform import axis2transformation
from s2u.utils.saver import get_mesh_pose_dict_from_world
from s2u.utils.visual import as_mesh
from s2u.utils.implicit import sample_iou_points_occ
from s2u.utils.io import write_data



def binary_occ(occ_list, idx):
    occ_fore = occ_list.pop(idx)
    occ_back = np.zeros_like(occ_fore)
    for o in occ_list:
        occ_back += o
    return occ_fore, occ_back


def sample_occ(sim, num_point, method, var=0.005):
    result_dict = get_mesh_pose_dict_from_world(sim.world, False)
    obj_name = str(sim.object_urdfs[sim.object_idx])
    obj_name = '/'.join(obj_name.split('/')[-4:-1])
    mesh_dict = {}
    whole_scene = trimesh.Scene()
    for k, v in result_dict.items():
        scene = trimesh.Scene()
        for mesh_path, scale, pose in v:
            mesh = trimesh.load(mesh_path)
            mesh.apply_scale(scale)
            mesh.apply_transform(pose)
            scene.add_geometry(mesh)
            whole_scene.add_geometry(mesh)
        mesh_dict[k] = as_mesh(scene)
    points_occ, occ_list = sample_iou_points_occ(mesh_dict.values(),
                                                      whole_scene.bounds,
                                                      num_point,
                                                      method,
                                                      var=var)
    return points_occ, occ_list

def sample_occ_binary(sim, mobile_links, num_point, method, var=0.005):
    result_dict = get_mesh_pose_dict_from_world(sim.world, False)
    new_dict = {'0_0': [], '0_1': []}
    obj_name = str(sim.object_urdfs[sim.object_idx])
    obj_name = '/'.join(obj_name.split('/')[-4:-1])
    whole_scene = trimesh.Scene()
    static_scene = trimesh.Scene()
    mobile_scene = trimesh.Scene()
    for k, v in result_dict.items():
        body_uid, link_index = k.split('_')
        link_index = int(link_index)
        if link_index in mobile_links:
            new_dict['0_1'] += v
        else:
            new_dict['0_0'] += v
        for mesh_path, scale, pose in v:
            if mesh_path.startswith('#'): # primitive
                mesh = trimesh.creation.box(extents=scale, transform=pose)
            else:
                mesh = trimesh.load(mesh_path)
                mesh.apply_scale(scale)
                mesh.apply_transform(pose)
            if link_index in mobile_links:
                mobile_scene.add_geometry(mesh)
            else:
                static_scene.add_geometry(mesh)
            whole_scene.add_geometry(mesh)
    static_mesh = as_mesh(static_scene)
    mobile_mesh = as_mesh(mobile_scene)
    points_occ, occ_list = sample_iou_points_occ((static_mesh, mobile_mesh),
                                                      whole_scene.bounds,
                                                      num_point,
                                                      method,
                                                      var=var)
    return points_occ, occ_list, new_dict


def main(args, obj_idx_list):
    sim = ArticulatedObjectManipulationSim(args.object_set,
                                           size=0.3,
                                           gui=args.sim_gui,
                                           global_scaling=args.global_scaling,
                                           dense_photo=args.dense_photo)
    for idx in obj_idx_list:
        
        sim.reset(idx, canonical=True)
        object_path = str(sim.object_urdfs[sim.object_idx])
        
        
        results = collect_observations(
            sim, args)
        
        for result in results:
            result['object_path'] = object_path
            write_data(args.root, result)
        print(f'Object {idx} finished! Write {len(results)} items!')


def get_limit(v, args):
    joint_type = v[2]
    # specify revolute angle range for shape2motion
    if joint_type == 0 and not args.is_syn:
        if args.pos_rot:
            lower_limit = 0
            range_lim = np.pi / 2
            higher_limit = np.pi / 2
        else:
            lower_limit = - np.pi / 4
            range_lim = np.pi / 2
            higher_limit = np.pi / 4
    else:
        lower_limit = v[8]
        higher_limit = v[9]
        range_lim = higher_limit - lower_limit
    return lower_limit, higher_limit, range_lim

def collect_observations(sim, args):
    if args.is_syn:
        joint_info = sim.get_joint_info_w_sub()
    else:
        joint_info = sim.get_joint_info()
    all_joints = list(joint_info.keys())
    results = []
    for joint_index in all_joints:
        for tmp_index in all_joints:
            if tmp_index != joint_index:
                if args.rand_state:
                    v = joint_info[tmp_index]
                    lower_limit, higher_limit, range_lim = get_limit(v[0], args)
                    start_state = np.random.uniform(lower_limit, higher_limit)
                    sim.set_joint_state(tmp_index, start_state)
                else:
                    sim.set_joint_state(tmp_index, 0)

        v = joint_info[joint_index]
        if args.is_syn:
            v = v[0]
        axis, moment = sim.get_joint_screw(joint_index)
        joint_type = v[2]
        # not set
        lower_limit, higher_limit, range_lim = get_limit(v, args)

        start_state = higher_limit - range_lim / 4
        end_state = lower_limit + range_lim / 4
        
        if np.random.uniform(0, 1) > 0.5:
            start_state, end_state = end_state, start_state
        
        sim.set_joint_state(joint_index, start_state)
        if args.is_syn:
            _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pc(6, joint_info[joint_index][1])
            start_p_occ, start_occ_list, start_mesh_pose_dict = sample_occ_binary(sim, joint_info[joint_index][1], args.num_point_occ, args.sample_method, args.occ_var)
        else:
            _, start_pc, start_seg_label, start_mesh_pose_dict = sim.acquire_segmented_pc(6, [joint_index])
            start_p_occ, start_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)
        # canonicalize start pc
        axis, moment = sim.get_joint_screw(joint_index)
        state_change = end_state - start_state
        if joint_type == 0:
            transformation = axis2transformation(axis, np.cross(axis, moment), state_change)
        else:
            transformation = np.eye(4)
            transformation[:3, 3] = axis * state_change
        
        mobile_start_pc = start_pc[start_seg_label].copy()
        rotated = transformation[:3, :3].dot(mobile_start_pc.T) + transformation[:3, [3]]
        rotated = rotated.T
        canonical_start_pc = start_pc.copy()
        canonical_start_pc[start_seg_label] = rotated

        sim.set_joint_state(joint_index, end_state)
        if args.is_syn:
            _, end_pc, end_seg_label, end_mesh_pose_dict = sim.acquire_segmented_pc(6, joint_info[joint_index][1])
            end_p_occ, end_occ_list, end_mesh_pose_dict = sample_occ_binary(sim, joint_info[joint_index][1], args.num_point_occ, args.sample_method, args.occ_var)
        else:
            _, end_pc, end_seg_label, end_mesh_pose_dict = sim.acquire_segmented_pc(6, [joint_index])
            end_p_occ, end_occ_list = sample_occ(sim, args.num_point_occ, args.sample_method, args.occ_var)
        # canonicalize end pc
        axis, moment = sim.get_joint_screw(joint_index)
        state_change = start_state - end_state
        if joint_type == 0:
            transformation = axis2transformation(axis, np.cross(axis, moment), state_change)
        else:
            transformation = np.eye(4)
            transformation[:3, 3] = axis * state_change
        mobile_end_pc = end_pc[end_seg_label].copy()
        rotated = transformation[:3, :3].dot(mobile_end_pc.T) + transformation[:3, [3]]
        rotated = rotated.T
        canonical_end_pc = end_pc.copy()
        canonical_end_pc[end_seg_label] = rotated

        result = {
                'pc_start': start_pc,
                'pc_start_end': canonical_start_pc,
                'pc_seg_start': start_seg_label,
                'pc_end': end_pc,
                'pc_end_start': canonical_end_pc,
                'pc_seg_end': end_seg_label,
                'state_start': start_state,
                'state_end': end_state,
                'screw_axis': axis,
                'screw_moment': moment,
                'joint_type': joint_type,
                'joint_index': 1 if args.is_syn else joint_index,
                'start_p_occ': start_p_occ, 
                'start_occ_list': start_occ_list, 
                'end_p_occ': end_p_occ, 
                'end_occ_list': end_occ_list,
                'start_mesh_pose_dict': start_mesh_pose_dict,
                'end_mesh_pose_dict': end_mesh_pose_dict
            }
        results.append(result)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--object-set", type=str)
    parser.add_argument("--sim-gui", action="store_true")
    parser.add_argument("--num-point-occ", type=int, default=100000)
    parser.add_argument("--occ-var", type=float, default=0.005)
    parser.add_argument("--sample-method", type=str, default='uniform')
    parser.add_argument("--rand-state", action="store_true", help='set static joints at random state')
    parser.add_argument("--global-scaling", type=float, default=0.5)
    parser.add_argument("--dense-photo", action="store_true")
    parser.add_argument("--mp", action="store_true", help='mulitprocess')
    parser.add_argument("--pos-rot", type=int, required=True)


    args = parser.parse_args()
    if 'syn' in args.object_set:
        args.is_syn = True
    else:
        args.is_syn = False
    
    num_proc = len(glob.glob(os.path.join('data/urdfs', args.object_set, '*', '*.urdf')))
    print(f'Number of objects: {num_proc}')
    (args.root / "scenes").mkdir(parents=True)

    if args.mp:
        pool = mp.get_context("spawn").Pool(processes=num_proc)
        for i in range(num_proc):
            pool.apply_async(func=main, args=(args, [i]))
        pool.close()
        pool.join()
    else:
        main(args, range(num_proc))
