"""
Scripted pick-and-place operator for the KTaskPlace family, batched over env slots on the GPU.

It plans from the true object and target poses: approach above the grasp point, descend, close, lift,
carry at a fixed height, lower to the placement height, release, retreat (air targets: move the object
to the goal point and hold). Commands are absolute tcp poses (top-down, yaw only) moving toward each
waypoint at a per-episode speed, so the recorded actions are what the policy imitates.

Quality q in [0, 1] per episode scales its mistakes, like transfer/gen_mixed.py on robosuite:
  grasp: xy offset ~ N(0, (q * 1 cm)^2), height ~ N(0, (q * 0.6 cm)^2), yaw ~ N(0, (q * 0.25 rad)^2),
         and with probability 0.3 q it closes the gripper early (up to 2 cm high);
  placement: xy offset ~ N(0, (q * 2.5 cm)^2), released up to q * 5 cm higher;
  every step: position jitter ~ N(0, (q * 2 mm)^2).
q = 0 is the clean expert. It never retries: a missed grasp still runs the rest of the plan.
"""

import math

import torch

from ktask.env import (BIN_FLOOR_TOP, OBJ_NAMES, OBJECTS, PLATFORM_TOP, ROBOT_ROOT)

APPROACH, DESCEND, CLOSE, LIFT, CARRY, LOWER, RELEASE, RETREAT, DONE = range(9)
PRE_H, CARRY_Z = 0.08, 0.17
N_CLOSE, N_RELEASE = 6, 6
YAW_STEP = 0.12


def topdown_matrix(phi):
    """tcp rotation pointing down with the fingers closing along (cos phi, sin phi, 0): columns
    [closing x approach, closing, approach] as in Panda.build_grasp_pose."""
    c, s = torch.cos(phi), torch.sin(phi)
    z, o = torch.zeros_like(phi), torch.ones_like(phi)
    return torch.stack([torch.stack([-s, c, z], -1), torch.stack([c, s, z], -1), torch.stack([z, z, -o], -1)], -2)


def wrap_half(x):
    """Wrap to [-pi/2, pi/2): the gripper is symmetric under a half turn."""
    return torch.remainder(x + math.pi / 2, math.pi) - math.pi / 2


class Expert:
    def __init__(self, env, q, gen):
        """env: the unwrapped KTaskPlaceEnv right after reset; q: (N,) quality; gen: torch.Generator on the device."""
        self.env = env
        dev = env.device
        N = env.num_envs
        self.N, self.dev = N, dev
        self.q = q.to(dev).float()
        r = lambda *s: torch.randn(*s, generator=gen, device=dev)
        u = lambda *s: torch.rand(*s, generator=gen, device=dev)
        q1 = self.q[:, None]
        self.grasp_off = torch.cat([r(N, 2) * q1 * 0.010, r(N, 1) * q1 * 0.006], 1)
        self.yaw_off = r(N) * self.q * 0.25
        self.early = (u(N) < 0.3 * self.q) * u(N) * 0.02            # close this far above the grasp point
        self.place_off = r(N, 2) * q1 * 0.025
        self.release_h = u(N) * self.q * 0.05
        self.jitter = self.q * 0.002
        self.v_max = 0.010 + 0.006 * u(N)
        self.gen = gen

        self.cls = env.obj_cls
        self.grasp_local = torch.tensor([OBJECTS[o]['grasp'] for o in OBJ_NAMES], device=dev)[self.cls]
        self.close_angle = torch.tensor([OBJECTS[o]['close'] for o in OBJ_NAMES], device=dev)[self.cls]
        sym = [OBJECTS[o]['sym'] for o in OBJ_NAMES]
        self.sym = torch.tensor([s if s is not None else 0.0 for s in sym], device=dev)[self.cls]
        self.n_sym = torch.tensor([round(2 * math.pi / s) if s is not None else 1 for s in sym], device=dev)[self.cls]
        self.root = torch.tensor(ROBOT_ROOT, device=dev)

        tcp = env.agent.tcp.pose
        self.cmd_p = tcp.p.clone()
        R = self._tcp_matrix()
        self.cmd_phi = torch.atan2(R[:, 1, 1], R[:, 0, 1])       # current closing direction
        self.grip = torch.ones(N, device=dev)
        self.phase = torch.zeros(N, dtype=torch.long, device=dev)
        self.phase_t = torch.zeros(N, dtype=torch.long, device=dev)
        self.k = None                                             # chosen grasp candidate per env
        self.phi_target = self.cmd_phi.clone()
        self.hold_off = torch.zeros(N, 3, device=dev)            # tcp minus object at the grasp
        self.wp = self.cmd_p.clone()
        self._plan_grasp(first=True)

    def _tcp_matrix(self):
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
        return quaternion_to_matrix(self.env.agent.tcp.pose.q)

    def _obj_yaw(self):
        from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
        R = quaternion_to_matrix(self.env.obj.pose.q)
        return torch.atan2(R[:, 1, 0], R[:, 0, 0])

    def _candidates(self):
        """Grasp points (N, 4, 3) and closing angles (N, 4) for each symmetric grasp (unused slots: inf cost)."""
        theta = self._obj_yaw()
        p_o = self.env.obj.pose.p
        ks = torch.arange(4, device=self.dev)
        th = theta[:, None] + ks[None] * self.sym[:, None]                        # (N, 4)
        gl = self.grasp_local
        gx = torch.cos(th) * gl[:, None, 0] - torch.sin(th) * gl[:, None, 1]
        gy = torch.sin(th) * gl[:, None, 0] + torch.cos(th) * gl[:, None, 1]
        pts = p_o[:, None] + torch.stack([gx, gy, gl[:, None, 2].expand_as(gx)], -1)
        phis = th + self.close_angle[:, None]
        valid = ks[None] < self.n_sym[:, None]
        return pts, phis, valid

    def _plan_grasp(self, first=False):
        """Grasp point and yaw from the object's current pose. The symmetric grasp (and the half turn of
        the gripper) is chosen once, at the start, for the least wrist rotation (ties: the grasp point
        nearest the robot); afterwards the plan follows the object."""
        pts, phis, valid = self._candidates()
        if first:
            diff = wrap_half(phis - self.cmd_phi[:, None]).abs()
            near = 5.0 * (pts[..., 0] - pts[..., 0].masked_fill(~valid, 1e9).min(1, keepdim=True).values)
            self.k = (diff + near).masked_fill(~valid, 1e9).argmin(1)
        idx = self.k[:, None]
        p = pts.gather(1, idx[..., None].expand(-1, 1, 3))[:, 0]
        phi = phis.gather(1, idx)[:, 0] + self.yaw_off
        round_obj = self.sym == 0                                  # any yaw works: keep the current one
        ref = self.cmd_phi if first else self.phi_target
        self.phi_target = torch.where(round_obj, ref, ref + wrap_half(phi - ref))
        g = p + self.grasp_off
        g[:, 2] = g[:, 2].clamp(min=0.010)
        self.grasp_p = g

    def _place_point(self):
        e = self.env
        t = e.tgt_type
        tp = e.target_point
        obj_z = torch.where(t == 0, e.rest_z + 0.005,
                            torch.where(t == 1, PLATFORM_TOP + e.rest_z + 0.008,
                                        torch.where(t == 2, BIN_FLOOR_TOP + e.rest_z + 0.02, tp[:, 2])))
        extra = torch.where(t == 3, torch.zeros_like(self.release_h), self.release_h)
        xy = tp[:, :2] + self.place_off
        return torch.cat([xy, (obj_z + extra)[:, None]], 1) + self.hold_off

    def act(self):
        """Advance the phase machine on the current state; returns the 10-D policy action (root frame)."""
        e = self.env
        tcp = e.agent.tcp.pose.p
        ph = self.phase
        self.phase_t += 1
        dist = lambda w: torch.linalg.norm(tcp - w, dim=1)

        # waypoints for the current phases
        in_grasp = (ph == APPROACH) | (ph == DESCEND)
        if in_grasp.any():
            self._plan_grasp()
        wp = self.wp.clone()
        above = self.grasp_p + torch.tensor([0, 0, PRE_H], device=self.dev)
        wp = torch.where((ph == APPROACH)[:, None], above, wp)
        wp = torch.where((ph == DESCEND)[:, None], self.grasp_p, wp)
        place = self._place_point()
        lift = torch.cat([tcp[:, :2], torch.full_like(tcp[:, :1], CARRY_Z)], 1)
        carry = torch.cat([place[:, :2], torch.full_like(tcp[:, :1], CARRY_Z)], 1)
        wp = torch.where((ph == CARRY)[:, None], carry, wp)
        wp = torch.where((ph == LOWER)[:, None], place, wp)
        self.wp = wp

        # transitions
        yaw_err = wrap_half(self.cmd_phi - self.phi_target).abs()
        nxt = ph.clone()
        nxt[(ph == APPROACH) & (dist(above) < 0.012) & (yaw_err < 0.08)] = DESCEND
        close_at = 0.006 + self.early
        nxt[(ph == DESCEND) & ((dist(self.grasp_p) < close_at) | (self.phase_t > 40))] = CLOSE
        nxt[(ph == CLOSE) & (self.phase_t >= N_CLOSE)] = LIFT
        nxt[(ph == LIFT) & ((tcp[:, 2] - CARRY_Z).abs() < 0.015)] = CARRY
        nxt[(ph == CARRY) & (torch.linalg.norm(tcp[:, :2] - carry[:, :2], dim=1) < 0.012)] = LOWER
        lowered = (ph == LOWER) & ((dist(place) < 0.008) | (self.phase_t > 40))
        nxt[lowered & (e.tgt_type == 3)] = DONE
        nxt[lowered & (e.tgt_type != 3)] = RELEASE
        nxt[(ph == RELEASE) & (self.phase_t >= N_RELEASE)] = RETREAT
        nxt[(ph == RETREAT) & ((((tcp[:, 2] - CARRY_Z).abs() < 0.02)) | (self.phase_t > 20))] = DONE
        changed = nxt != ph
        # remember where the object sits in the hand when the gripper has closed
        grabbed = changed & (nxt == LIFT)
        if grabbed.any():
            self.hold_off[grabbed] = (tcp - e.obj.pose.p)[grabbed]
        self.phase_t[changed] = 0
        self.phase = nxt
        ph = nxt
        # waypoint for phases entered just now
        lift_now = changed & ((ph == LIFT) | (ph == RETREAT))
        self.wp[lift_now] = lift[lift_now]
        if ph.eq(CARRY).any() or ph.eq(LOWER).any():
            place = self._place_point()
            carry = torch.cat([place[:, :2], torch.full_like(tcp[:, :1], CARRY_Z)], 1)
            self.wp = torch.where((ph == CARRY)[:, None], carry, self.wp)
            self.wp = torch.where((ph == LOWER)[:, None], place, self.wp)
        hold = (ph == CLOSE) | (ph == RELEASE) | (ph == DONE)

        # commands: move toward the waypoint at this episode's speed, rotate toward the grasp yaw
        delta = self.wp - self.cmd_p
        n = torch.linalg.norm(delta, dim=1, keepdim=True).clamp(min=1e-9)
        step = delta * torch.clamp(self.v_max[:, None] / n, max=1.0)
        self.cmd_p = torch.where(hold[:, None], self.cmd_p, self.cmd_p + step)
        dphi = wrap_half(self.phi_target - self.cmd_phi).clamp(-YAW_STEP, YAW_STEP)
        rotating = (ph == APPROACH) | (ph == DESCEND)
        self.cmd_phi = torch.where(rotating, self.cmd_phi + dphi, self.cmd_phi)
        self.grip = torch.where((ph >= CLOSE) & (ph < RELEASE), -torch.ones_like(self.grip),
                                torch.where(ph >= RELEASE, torch.ones_like(self.grip), self.grip))
        self.grip = torch.where((ph == DONE) & (e.tgt_type == 3), -torch.ones_like(self.grip), self.grip)
        jit = torch.randn(self.N, 3, generator=self.gen, device=self.dev) * self.jitter[:, None]
        pos = self.cmd_p + jit - self.root
        from ktask.env import rot6d
        return torch.cat([pos, rot6d(topdown_matrix(self.cmd_phi)), self.grip[:, None]], 1)

    @property
    def done(self):
        return self.phase == DONE
