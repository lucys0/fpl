"""
A family of 32 pick-and-place tasks for the K-task transfer experiments: 8 objects, each needing its own
grasp, x 4 targets, each needing its own placement, on the ManiSkill3 Panda tabletop.

Task id = 4 * object + target. Each env slot gets one task when the env is built (`task_ids`, one per
slot). Episodes randomize the object pose (right of the robot, y < 0) and the target position (left, y > 0).

Policy observation (40-D, robot root frame; policy_obs): tcp position (3) and rotation (6D, 6), gripper
width (1), object position (3) and rotation (6), object minus tcp (3), target point (3), target point minus
object (3), object one-hot (8), target one-hot (4).
Policy action (10-D): absolute tcp target position (3) and rotation (6D, 6), gripper (+1 open, -1 closed),
executed by the pd_ee_pose controller (to_env_action).

Success: goal - the object rests on the table within 3 cm (xy) of the goal spot; platform - it rests on the
platform's top; bin - it rests inside the bin; air - its center is within 2.5 cm of the goal point (held).
The three placements also require the object to be static and released.
"""

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.geometry.rotation_conversions import (matrix_to_euler_angles, quaternion_to_matrix)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

ROBOT_ROOT = (-0.615, 0.0, 0.0)

# shapes: (kind, local position, size) with box size = half extents, cyl size = (radius, half height), upright
# grasp: tcp point in the object frame; close: closing direction in the object frame (angle from its x axis);
# sym: rotation symmetry of the grasp (None: any yaw)
Y = np.pi / 2
OBJECTS = {
    'cube': dict(shapes=[('box', (0, 0, 0), (0.02, 0.02, 0.02))], rest_z=0.02, grasp=(0, 0, 0), close=Y, sym=np.pi / 2),
    'bar': dict(shapes=[('box', (0, 0, 0), (0.045, 0.012, 0.015))], rest_z=0.015, grasp=(0, 0, 0), close=Y, sym=np.pi),
    'cylinder': dict(shapes=[('cyl', (0, 0, 0), (0.02, 0.03))], rest_z=0.03, grasp=(0, 0, 0), close=Y, sym=None),
    'puck': dict(shapes=[('cyl', (0, 0, 0), (0.03, 0.01))], rest_z=0.01, grasp=(0, 0, 0.004), close=Y, sym=None),
    'tall': dict(shapes=[('box', (0, 0, 0), (0.012, 0.012, 0.045))], rest_z=0.045, grasp=(0, 0, 0.02), close=Y,
                 sym=np.pi / 2),
    'plate': dict(shapes=[('box', (0, 0, 0), (0.035, 0.022, 0.008))], rest_z=0.008, grasp=(0, 0, 0.006), close=Y,
                  sym=np.pi),
    # T: crossbar along x, stem along y; grasped across the stem just below the crossbar
    'tee': dict(shapes=[('box', (0, 0.03, 0), (0.05, 0.012, 0.015)), ('box', (0, -0.012, 0), (0.012, 0.03, 0.015))],
                rest_z=0.015, grasp=(0, 0, 0), close=0.0, sym=np.pi),
    # square frame (9 cm), grasped across one wall: a hole wide enough for the inner finger
    'ring': dict(shapes=[('box', (0, 0.039, 0), (0.045, 0.006, 0.012)), ('box', (0, -0.039, 0), (0.045, 0.006, 0.012)),
                         ('box', (0.039, 0, 0), (0.006, 0.033, 0.012)), ('box', (-0.039, 0, 0), (0.006, 0.033, 0.012))],
                 rest_z=0.012, grasp=(0, 0.039, 0), close=Y, sym=np.pi / 2),
}
OBJ_NAMES = list(OBJECTS)
TGT_NAMES = ['goal', 'platform', 'bin', 'air']
N_OBJ, N_TGT = len(OBJ_NAMES), len(TGT_NAMES)
N_TASKS = N_OBJ * N_TGT
OBS_DIM, ACT_DIM = 40, 10
COLORS = [(0.85, 0.2, 0.2), (0.2, 0.55, 0.85), (0.25, 0.7, 0.3), (0.9, 0.6, 0.1),
          (0.6, 0.3, 0.75), (0.15, 0.7, 0.7), (0.8, 0.4, 0.55), (0.6, 0.38, 0.12)]

PLATFORM_HALF = (0.05, 0.05, 0.03)          # top at z = 0.06
PLATFORM_TOP = 2 * PLATFORM_HALF[2]
BIN_INNER, BIN_WALL, BIN_FLOOR, BIN_WALL_H = 0.065, 0.003, 0.004, 0.02   # half sizes
BIN_FLOOR_TOP = 2 * BIN_FLOOR
BIN_RIM = BIN_FLOOR_TOP + 2 * BIN_WALL_H     # 0.048
GOAL_XY_TOL, AIR_TOL = 0.03, 0.025
OBJ_REGION = ((-0.08, 0.04), (-0.20, -0.07))     # x range, y range (world)
TGT_REGION = ((-0.08, 0.04), (0.08, 0.20))
AIR_Z = (0.10, 0.22)


def task_id(obj, tgt):
    return OBJ_NAMES.index(obj) * N_TGT + TGT_NAMES.index(tgt)


def task_name(t):
    return f'{OBJ_NAMES[t // N_TGT]}-{TGT_NAMES[t % N_TGT]}'


def rot6d(R):
    """First two rows of the rotation matrix (pytorch3d's convention)."""
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def rot6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack((b1, b2, torch.cross(b1, b2, dim=-1)), dim=-2)


def to_env_action(a10):
    """10-D policy action -> pd_ee_pose action (root-frame position, XYZ Euler angles, gripper)."""
    euler = matrix_to_euler_angles(rot6d_to_matrix(a10[..., 3:9]), 'XYZ')
    return torch.cat([a10[..., :3], euler, a10[..., 9:10].clamp(-1, 1)], dim=-1)


def _add_shapes(builder, shapes, color, collision=True):
    mat = sapien.render.RenderMaterial(base_color=[*color, 1.0])
    for kind, p, size in shapes:
        if kind == 'box':
            pose = sapien.Pose(p=p)
            if collision:
                builder.add_box_collision(pose, half_size=size)
            builder.add_box_visual(pose, half_size=size, material=mat)
        else:      # upright cylinder: SAPIEN cylinders lie along x
            pose = sapien.Pose(p=p, q=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
            if collision:
                builder.add_cylinder_collision(pose, radius=size[0], half_length=size[1])
            builder.add_cylinder_visual(pose, radius=size[0], half_length=size[1], material=mat)


@register_env('KTaskPlace-v0', max_episode_steps=100_000)
class KTaskPlaceEnv(BaseEnv):
    SUPPORTED_ROBOTS = ['panda']

    def __init__(self, *args, task_ids=None, robot_uids='panda', robot_init_qpos_noise=0.02, **kwargs):
        self.task_ids_init = list(task_ids) if task_ids is not None else None
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, reconfiguration_freq=0, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2 ** 25,
                                                           max_rigid_patch_count=2 ** 19,
                                                           max_rigid_contact_count=2 ** 21))

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        from mani_skill.sensors.camera import CameraConfig
        from mani_skill.utils import sapien_utils
        pose = sapien_utils.look_at([0.28, -0.48, 0.36], [-0.04, 0.0, 0.04])
        return CameraConfig('render_camera', pose, 512, 512, 0.9, 0.01, 100)

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(p=list(ROBOT_ROOT)))

    def _load_scene(self, options):
        n = self.num_envs
        ids = self.task_ids_init if self.task_ids_init is not None else [i % N_TASKS for i in range(n)]
        assert len(ids) == n, (len(ids), n)
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        # one actor per object type and per target type, spanning the slots that use it, then merged in slot
        # order (Actor.merge keeps build order; set_pose writes rows in slot order)
        ids_t = torch.tensor(ids)
        objs, tgts = [], []
        for c, o in enumerate(OBJ_NAMES):
            slots = (ids_t // N_TGT == c).nonzero().flatten().tolist()
            if slots:
                b = self.scene.create_actor_builder()
                _add_shapes(b, OBJECTS[o]['shapes'], COLORS[c])
                b.initial_pose = sapien.Pose(p=[0, -0.15, OBJECTS[o]['rest_z']])
                b.set_scene_idxs(slots)
                objs.append(b.build(name=f'obj-{o}'))
        for g, kind in enumerate(TGT_NAMES):
            slots = (ids_t % N_TGT == g).nonzero().flatten().tolist()
            if slots:
                tgts.append(self._build_target(kind, slots))
        for a in objs + tgts:
            self.remove_from_state_dict_registry(a)
        self.obj = self._merge_in_slot_order(objs, 'obj')
        self.target = self._merge_in_slot_order(tgts, 'target')
        self.add_to_state_dict_registry(self.obj)
        self.add_to_state_dict_registry(self.target)

        dev = self.device
        self.task_ids = torch.tensor(ids, device=dev)
        self.obj_cls, self.tgt_type = self.task_ids // N_TGT, self.task_ids % N_TGT
        self.obj_onehot = torch.nn.functional.one_hot(self.obj_cls, N_OBJ).float()
        self.tgt_onehot = torch.nn.functional.one_hot(self.tgt_type, N_TGT).float()
        self.rest_z = torch.tensor([OBJECTS[o]['rest_z'] for o in OBJ_NAMES], device=dev)[self.obj_cls]
        self.target_point = torch.zeros(n, 3, device=dev)

    def _merge_in_slot_order(self, actors, name):
        ents, idxs, poses = [], [], []
        for a in actors:
            ents += a._objs
            idxs.append(a._scene_idxs)
            rp = a.initial_pose.raw_pose
            poses.append(rp.expand(len(a._objs), -1) if rp.shape[0] == 1 else rp)
        idxs = torch.cat(idxs)
        order = torch.argsort(idxs)
        assert torch.equal(idxs[order].cpu(), torch.arange(self.num_envs)), 'each slot needs exactly one actor'
        merged = Actor.create_from_entities([ents[i] for i in order.tolist()], self.scene, idxs[order])
        merged.name = name
        merged.initial_pose = Pose.create(torch.vstack(poses)[order.to(torch.vstack(poses).device)])
        merged.merged = True
        self.scene.actor_views[name] = merged
        return merged

    def _build_target(self, kind, slots):
        b: ActorBuilder = self.scene.create_actor_builder()
        grey = sapien.render.RenderMaterial(base_color=[0.55, 0.5, 0.45, 1])
        if kind == 'platform':
            b.add_box_collision(half_size=PLATFORM_HALF)
            b.add_box_visual(half_size=PLATFORM_HALF, material=grey)
        elif kind == 'bin':
            c, w, h = BIN_INNER + BIN_WALL, BIN_WALL, BIN_WALL_H
            parts = [((0, 0, BIN_FLOOR), (c + w, c + w, BIN_FLOOR))]
            for s in (-1, 1):
                parts.append(((s * c, 0, BIN_FLOOR_TOP + h), (w, c + w, h)))
                parts.append(((0, s * c, BIN_FLOOR_TOP + h), (c - w, w, h)))
            for p, hs in parts:
                b.add_box_collision(sapien.Pose(p=p), half_size=hs)
                b.add_box_visual(sapien.Pose(p=p), half_size=hs, material=grey)
        elif kind == 'goal':       # a flat marker, no collision
            b.add_cylinder_visual(sapien.Pose(q=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0]), radius=GOAL_XY_TOL,
                                  half_length=0.002, material=sapien.render.RenderMaterial(base_color=[0.1, 0.9, 0.1, 1]))
        else:                      # air: a small marker, no collision
            b.add_sphere_visual(radius=0.008, material=sapien.render.RenderMaterial(base_color=[0.2, 0.8, 0.2, 1]))
        b.initial_pose = sapien.Pose(p=[0, 0.15, 0.0])
        b.set_scene_idxs(slots)
        return b.build_kinematic(name=f'target-{kind}')

    def _initialize_episode(self, env_idx, options):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            rng = self._batched_episode_rng[env_idx]
            T = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.device)
            (x0, x1), (y0, y1) = OBJ_REGION
            xy = T(np.stack([rng.uniform(x0, x1), rng.uniform(y0, y1)], -1))
            yaw = T(rng.uniform(0, 2 * np.pi))
            p = torch.cat([xy, self.rest_z[env_idx, None]], 1)
            q = torch.stack([torch.cos(yaw / 2), torch.zeros(b), torch.zeros(b), torch.sin(yaw / 2)], 1)
            self.obj.set_pose(Pose.create_from_pq(p=p, q=q))
            (x0, x1), (y0, y1) = TGT_REGION
            txy = T(np.stack([rng.uniform(x0, x1), rng.uniform(y0, y1)], -1))
            air_z = T(rng.uniform(*AIR_Z))
            kind = self.tgt_type[env_idx]
            body_z = torch.where(kind == 1, torch.tensor(PLATFORM_HALF[2]), torch.tensor(0.0))
            body_z = torch.where(kind == 3, air_z, body_z)
            self.target.set_pose(Pose.create_from_pq(p=torch.cat([txy, body_z[:, None]], 1)))
            point_z = torch.where(kind == 1, torch.tensor(PLATFORM_TOP),
                                  torch.where(kind == 2, torch.tensor(BIN_FLOOR_TOP), torch.tensor(0.0)))
            point_z = torch.where(kind == 3, air_z, point_z)
            self.target_point[env_idx] = torch.cat([txy, point_z[:, None]], 1)

    def evaluate(self):
        obj_p = self.obj.pose.p
        grasped = self.agent.is_grasping(self.obj)
        static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        d = obj_p - self.target_point
        dxy_max = d[:, :2].abs().max(1).values
        on_goal = (torch.linalg.norm(d[:, :2], dim=1) <= GOAL_XY_TOL) & (obj_p[:, 2] <= self.rest_z + 0.01)
        on_platform = (dxy_max <= PLATFORM_HALF[0]) & (d[:, 2] > 0.0) & (d[:, 2] < 0.1)
        in_bin = (dxy_max <= BIN_INNER) & (obj_p[:, 2] < BIN_RIM + 0.01)
        at_air = torch.linalg.norm(d, dim=1) <= AIR_TOL
        t = self.tgt_type
        placed = torch.where(t == 0, on_goal, torch.where(t == 1, on_platform, in_bin)) & static & ~grasped
        success = torch.where(t == 3, at_air, placed)
        lifted = obj_p[:, 2] > self.rest_z + 0.03
        return dict(success=success, grasped=grasped, lifted=lifted, fallen=obj_p[:, 2] < -0.05)

    def _get_obs_extra(self, info):
        return dict()

    def policy_obs(self):
        root = self.agent.robot.pose.p
        tcp = self.agent.tcp.pose
        tcp_p = tcp.p - root
        width = self.agent.robot.get_qpos()[:, -2:].sum(1, keepdim=True)
        obj = self.obj.pose
        obj_p = obj.p - root
        tgt = self.target_point - root
        return torch.cat([tcp_p, rot6d(quaternion_to_matrix(tcp.q)), width, obj_p, rot6d(quaternion_to_matrix(obj.q)),
                          obj_p - tcp_p, tgt, tgt - obj_p, self.obj_onehot, self.tgt_onehot], 1)

    def compute_dense_reward(self, obs, action, info):
        return info['success'].float()

    def compute_normalized_dense_reward(self, obs, action, info):
        return info['success'].float()
