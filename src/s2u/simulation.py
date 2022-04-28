import json
import trimesh
from collections import OrderedDict
from pathlib import Path
import numpy as np
from treelib import Node, Tree
import pybullet

from s2u.utils import btsim, workspace_lines
from s2u.perception import *
from s2u.utils.transform import Transform, Rotation, get_transform
from s2u.utils.saver import get_mesh_pose_dict_from_world
from s2u.utils.visual import as_mesh


class ArticulatedObjectManipulationSim(object):
    def __init__(self, object_set, size=0.3, global_scaling=0.5, gui=True, seed=None, add_noise=False, name_list=None, dense_photo=False, urdf_root=None):

        self.size = size
        if urdf_root is None:
            self.urdf_root = Path('data/urdfs')
        else:
            self.urdf_root = Path(urdf_root)
        self.object_set = object_set
        self.discover_objects(name_list)

        self.gui = gui
        self.global_scaling = global_scaling
        self.dense_photo = dense_photo

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui)
        #intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0)
        intrinsic = CameraIntrinsic(1280, 960, 1080.0, 1080.0, 640.0, 480.0)
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)
        
    def discover_objects(self, name_list=None):
        root = self.urdf_root / self.object_set
        self.object_urdfs = []
        self.object_bbox = []
        if name_list is None:
            for f in root.iterdir():
                if not f.is_dir():
                    continue
                for f_tmp in f.iterdir():
                    if f_tmp.suffix == ".urdf":
                        self.object_urdfs.append(f_tmp)
                    if f_tmp.name == "bounding_box.json":
                        with open(f_tmp, 'r') as f:
                            d_tmp = json.load(f)
                        self.object_bbox.append(np.array((d_tmp['min'], d_tmp['max'])))
        else:
            for name in name_list:
                f = root / name
                if not f.is_dir():
                    continue
                for f_tmp in f.iterdir():
                    if f_tmp.suffix == ".urdf":
                        self.object_urdfs.append(f_tmp)
                    if f_tmp.name == "bounding_box.json":
                        with open(f_tmp, 'r') as f:
                            d_tmp = json.load(f)
                        self.object_bbox.append(np.array((d_tmp['min'], d_tmp['max'])))
                

    def reset(self, index=None, canonical=False):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=self.size * 2,
                cameraYaw=-60,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0],
            )

        table_height = 0.05
#         self.place_table(table_height)
        self.generate_scene(table_height, index, canonical)

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [self.size / 2, self.size / 2, height])
        self.world.load_urdf(urdf, pose, scale=self.size)

    def generate_scene(self, table_height, index=None, canonical=False):
        # drop objects
        if index is None:
            idx = self.rng.choice(np.arange(len(self.object_urdfs)), size=1)[0]
        else:
            idx = index
            
        self.object_idx = idx
        urdf = self.object_urdfs[idx]
        bbox = self.object_bbox[idx]
        obj_scale = bbox[1] - bbox[0]
        obj_scale = np.sqrt((obj_scale ** 2).sum())
        # import pdb; pdb.set_trace()
        if canonical:
            scale = self.global_scaling * self.size / obj_scale
            new_bbox = bbox * scale
            angle = 0
            rotation = Rotation.identity()
            xy = np.ones(2) * self.size / 2.0
            z = self.size / 2 - new_bbox[:, 2].mean()
            pose = Transform(rotation, np.r_[xy, z])
        else:
            scale = self.global_scaling * self.rng.uniform(0.8, 1.0) * self.size / obj_scale
            new_bbox = bbox * scale
            
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            #xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            xy = np.ones(2) * self.size / 2.0
            z = self.size / 2 - new_bbox[:, 2].mean()
            pose = Transform(rotation, np.r_[xy, z])
        
        self.object = self.world.load_urdf(urdf, pose, scale=scale, useFixedBase=True)
        # self.wait_for_objects_to_rest(timeout=1.0)
        
    def get_joint_info(self):
        num_joints = self.world.p.getNumJoints(self.object.uid)
        joint_info = OrderedDict()
        for joint_idx in range(num_joints):
            v = self.world.p.getJointInfo(self.object.uid, joint_idx)
            if v[2] in [0, 1]: # not revoluted or prsimatic
                joint_info[joint_idx] = v
        return joint_info

    def get_joint_info_w_sub(self):
        tree = Tree()
        tree.create_node('base', -1)
        joint_info = OrderedDict()
        num_joints = self.world.p.getNumJoints(self.object.uid)
        mobile_roots = []
        for joint_idx in range(num_joints):
            v = self.world.p.getJointInfo(self.object.uid, joint_idx)
            parent = v[-1]
            joint_name = v[1].decode('UTF-8')
            tree.create_node(joint_name, v[0], parent=parent, data=v)
            if v[2] in [0, 1]:
                mobile_roots.append(v[0])
        for node in mobile_roots:
            sub_nodes = tuple(tree.expand_tree(node))
            joint_info[node] = (tree.get_node(node).data, sub_nodes)
        return joint_info
    
    def set_joint_state(self, joint_index, target_value):
        self.world.p.resetJointState(self.object.uid, joint_index, target_value)
        
    def get_joint_screw(self, joint_index):
        joint_info_dict = self.get_joint_info()
        v = joint_info_dict[joint_index]
        joint_type = v[2]
        joint_axis = v[-4]
        joint_pos_parent = v[-3]
        joint_ori_parent = v[-2]
        parent_index = v[-1]
        if parent_index == -1: # baselink
            parent_link_state = self.world.p.getBasePositionAndOrientation(self.object.uid)
        else:
            parent_link_state = self.world.p.getLinkState(self.object.uid, parent_index)
        parent_link_trans = get_transform(parent_link_state[0], parent_link_state[1])
        relative_trans = get_transform(joint_pos_parent, joint_ori_parent)
        axis_trans = parent_link_trans * relative_trans
        axis_global = axis_trans.rotation.as_matrix().dot(joint_axis)
        axis_global /= np.sqrt(np.sum(axis_global ** 2))
        point_on_axis = axis_trans.translation
        moment = np.cross(point_on_axis, axis_global)
        return axis_global, moment
        
    def get_joint_state(self, joint_index):
        return self.world.p.getJointState(self.object.uid, joint_index)

    def acquire_tsdf(self, n, N=None):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.
        If N is None, the n viewpoints are equally distributed on circular trajectory.
        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        
        mesh_pose_dict = get_mesh_pose_dict_from_world(self.world, False)
        # tsdf = TSDFVolume(self.size, 40)
        high_res_tsdf = TSDFVolume(self.size, 120)

        origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, self.size / 2])
        r = 1.2 * self.size
        extrinsics = []

        if self.dense_photo:

            theta = np.pi / 8.0
            N = N if N else n
            phi_list = 2.0 * np.pi * np.arange(n) / N
            extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

            theta = np.pi / 4.0
            N = N if N else n
            phi_list = 2.0 * np.pi * np.arange(n) / N + 2.0 * np.pi / (N * 3)
            extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

            theta = np.pi / 2.0
            N = N if N else n
            phi_list = 2.0 * np.pi * np.arange(n) / N + 4.0 * np.pi / (N * 3)
            extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]
        else:
            theta = np.pi / 4.0
            N = N if N else n
            phi_list = 2.0 * np.pi * np.arange(n) / N
            extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]
        
        timing = 0.0
        depth_imgs = []
        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)[1]
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            depth_imgs.append(depth_img)
        pc = high_res_tsdf.get_cloud()
        pc = np.asarray(pc.points)
        
        return depth_imgs, pc, mesh_pose_dict


    def acquire_segmented_pc(self, n, mobile_links, N=None):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.
        If N is None, the n viewpoints are equally distributed on circular trajectory.
        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        
        mesh_pose_dict = get_mesh_pose_dict_from_world(self.world, False)
        # tsdf = TSDFVolume(self.size, 40)
        high_res_tsdf = TSDFVolume(self.size, 120, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, self.size / 2])
        r = 1.2 * self.size
        extrinsics = []

        if self.dense_photo:

            theta = np.pi / 8.0
            N = N if N else n
            phi_list = 2.0 * np.pi * np.arange(n) / N
            extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

            theta = np.pi / 4.0
            N = N if N else n
            phi_list = 2.0 * np.pi * np.arange(n) / N + 2.0 * np.pi / (N * 3)
            extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]

            theta = np.pi / 2.0
            N = N if N else n
            phi_list = 2.0 * np.pi * np.arange(n) / N + 4.0 * np.pi / (N * 3)
            extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]
        else:
            theta = np.pi / 4.0
            N = N if N else n
            phi_list = 2.0 * np.pi * np.arange(n) / N
            extrinsics += [camera_on_sphere(origin, r, theta, phi) for phi in phi_list]
        
        timing = 0.0
        depth_imgs = []
        for extrinsic in extrinsics:
            rgb_img, depth_img, (seg_uid, seg_link) = self.camera.render(extrinsic, flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
            seg_uid += 1
            seg_link_mobile = np.zeros_like(seg_link)
            for l in mobile_links:
                seg_link_mobile += seg_link == l
            seg_rgb = np.logical_and(seg_uid.astype(bool), seg_link_mobile)
            seg_rgb = (seg_rgb.astype(np.float32) * 255).astype(np.uint8)
            seg_rgb = np.stack([seg_rgb] * 3, axis=-1)
            # import pdb; pdb.set_trace()
            high_res_tsdf.integrate_rgb(depth_img, seg_rgb, self.camera.intrinsic, extrinsic)
            depth_imgs.append(depth_img)
        tmp = high_res_tsdf._volume.extract_point_cloud()
        pc = np.asarray(tmp.points)
        colors = np.asarray(tmp.colors)
        colors = np.mean(colors, axis=1)
        seg_mask = colors > 0.5
        return depth_imgs, pc, seg_mask, mesh_pose_dict

    def acquire_link_pc(self, link, num_points=2048):
        result_dict = get_mesh_pose_dict_from_world(self.world, False)
        scene = trimesh.Scene()
        for mesh_path, scale, pose in result_dict[f'0_{link}']:
            if mesh_path.startswith('#'): # primitive
                mesh = trimesh.creation.box(extents=scale, transform=pose)
            else:
                mesh = trimesh.load(mesh_path)
                mesh.apply_scale(scale)
                mesh.apply_transform(pose)
            scene.add_geometry(mesh)
        scene_mesh = as_mesh(scene)
        pc, _ = trimesh.sample.sample_surface(scene_mesh, num_points)
        return pc
    
    def acquire_full_pc(self, num_points=2048):
        result_dict = get_mesh_pose_dict_from_world(self.world, False)
        scene = trimesh.Scene()
        for k, v in result_dict.items():
            for mesh_path, scale, pose in v:
                if mesh_path.startswith('#'): # primitive
                    mesh = trimesh.creation.box(extents=scale, transform=pose)
                else:
                    mesh = trimesh.load(mesh_path)
                    mesh.apply_scale(scale)
                    mesh.apply_transform(pose)
                scene.add_geometry(mesh)
        scene_mesh = as_mesh(scene)
        pc, _ = trimesh.sample.sample_surface(scene_mesh, num_points)
        return pc, result_dict

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break