"""
A second, more different task pair: A = the square nut onto the square (left) peg, as before;
B = the ROUND nut onto the round (right) peg - a different object (an octagonal ring whose handle
sits 6 cm from its center instead of 5.4 cm), a different grasp and a different peg.

robosuite keeps both nuts in the scene and parks the unused one at (10, 10, 10). The
NutAssemblyRound variant observes the round nut with the same 23-D layout as the square task
('object' = the active nut's pose and its pose relative to the gripper), so one policy reads both;
the task is the goal input, as before (round = B's slot).
"""

import copy

import numpy as np

from diffusion_policy.env.robomimic.robomimic_twopeg_lowdim_wrapper import (_BASE_RESET_STATE,
                                                                           RobomimicTwoPegLowdimWrapper)

ROUND_HANDLE = 0.076        # handle grasp point from the nut center (square nut: 0.07)
RING_WALL = -0.04245        # center of the ring wall opposite the handle, along the nut's x axis
GRASP = 'ring'              # how the round-nut demos grasp: 'ring' (default) or 'handle'

# Grasping the handle, as for the square nut, makes the round-nut task almost the same motor program:
# a policy trained only on square-nut data (base, goal B) already puts the round nut on the round peg
# 5 times in 6. The demos therefore grasp the far side of the ring instead: a grasp point 12 cm away along
# the nut, the wrist turned 90 degrees, and a different offset between gripper and nut for insertion.
SQUARE, ROUND = slice(10, 17), slice(17, 24)       # free-joint qpos (pos, quat) of each nut in the sim state
PARKED = np.array([10., 10., 10., 1., 0., 0., 0.])


def round_state(state):
    """A square-task reset state with the round nut where the square nut was, and the square nut parked."""
    s = state.copy()
    s[ROUND] = s[SQUARE]
    s[SQUARE] = PARKED
    return s


def round_env_meta(env_meta):
    meta = copy.deepcopy(env_meta)
    meta['env_name'] = 'NutAssemblyRound'
    return meta


class RoundNutTwoPegWrapper(RobomimicTwoPegLowdimWrapper):
    """RobomimicTwoPegLowdimWrapper for the round nut (same start distribution, success on either peg)."""
    nut_name = 'RoundNut'

    def _make_init_state(self, rng=None):
        return round_state(super()._make_init_state(rng))

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        rs = self.env.env
        nut_pos = rs.sim.data.body_xpos[rs.obj_body_id[self.nut_name]]
        if rs.on_peg(nut_pos, 0) or rs.on_peg(nut_pos, 1):
            reward = 1.0
        return obs, reward, done, info


# ---------------------------------------------------------------- scripted demos (gen_mixed.py)

def collector_round_classes(collector):
    """Round-nut versions of the scripted collector's env wrapper and side-grasp policy."""

    class RoundMultimodalWrapper(collector.MultimodalSquareLowdimWrapper):
        def reset(self):
            super().reset()                                  # draws the start position, resets the square task
            self.last_reset_state = round_state(self.last_reset_state)
            self.env.env.reset_to({'states': self.last_reset_state})
            return self.get_observation()

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            rs = self._get_robosuite_env()
            nut_pos = rs.sim.data.body_xpos[rs.obj_body_id['RoundNut']]
            if rs.on_peg(nut_pos, 0) or rs.on_peg(nut_pos, 1):
                self.rew = 1
            return obs, (1 if self.rew == 1 else 0), done, info

    class RoundSideScriptedPolicy(collector.SquareSideScriptedPolicy):
        """SquareSideScriptedPolicy.generate_trajectory for the round nut: same phases (and the same
        120-point structure gen_mixed.py perturbs), with the round nut's grasp point (GRASP)."""

        def generate_trajectory(self, mode='middle'):
            R = collector.R
            rs = self._get_robosuite_env()
            self.nut_pos = rs.sim.data.body_xpos[rs.obj_body_id['RoundNut']]
            self.nut_rot = R.from_matrix(rs.sim.data.body_xmat[rs.obj_body_id['RoundNut']].reshape(3, 3)).as_euler('xyz')[2]
            peg_pos = np.array(rs.sim.data.body_xpos[rs.peg1_body_id if self.target_peg == 'left' else rs.peg2_body_id])
            if self.nut_rot > np.pi / 2:
                pick_rot = self.nut_rot - np.pi
            elif self.nut_rot < -np.pi / 2:
                pick_rot = np.pi + self.nut_rot
            else:
                pick_rot = self.nut_rot
            # handle: fingers across the handle, nut's x axis turned to world +y for insertion (as the square
            # nut); ring: fingers across the far wall (wrist turned 90 degrees), nut's x axis kept along world +x
            r, yaw_off, place_off = ((ROUND_HANDLE, 0.0, np.array([0., ROUND_HANDLE])) if GRASP == 'handle' else
                                     (RING_WALL, -3.14159 / 2, np.array([RING_WALL, 0.])))
            grasp_pos = self.nut_pos.copy()
            grasp_pos[2] -= 0.062
            grasp_pos[1] += r * np.sin(self.nut_rot)
            grasp_pos[0] += r * np.cos(self.nut_rot)
            above_nut_pos = grasp_pos.copy()
            above_nut_pos[2] += .2
            above_peg_pos = peg_pos.copy()
            above_peg_pos[2] += .25
            above_peg_pos[:2] += place_off
            insert_pos = peg_pos.copy()
            insert_pos[2] += 0.05
            insert_pos[:2] += place_off
            v = self.noise_level
            curved_descend = self.generate_curved_trajectory(above_nut_pos, grasp_pos, mode, v, 30)
            curved_lift = self.generate_curved_trajectory(grasp_pos, above_nut_pos, mode, v, 30)
            curved_transit = self.generate_curved_trajectory(above_nut_pos, above_peg_pos, mode, v, 30)
            curved_insert = self.generate_curved_trajectory(above_peg_pos, insert_pos, mode, 0.0, 25)
            angle, sf = 0, self.speed_factor
            grip = lambda yaw, g: np.array([0., 3.14159, yaw, g])
            pick = 3.14159 / 2 - pick_rot + yaw_off
            t = int(40 * sf)
            traj = [{"t": t, "action": np.concatenate([above_nut_pos, grip(pick, -1)])}]
            t += int(10 * sf)
            traj += [{"t": t + i, "action": np.concatenate([p, grip(pick, -1)])} for i, p in enumerate(curved_descend)]
            t += len(curved_descend) - 1
            t += int(5 * sf)
            traj.append({"t": t, "action": np.concatenate([grasp_pos, grip(pick, 1.)])})
            t += int(15 * sf)
            traj += [{"t": t + i, "action": np.concatenate([p, grip(pick, 1.)])} for i, p in enumerate(curved_lift)]
            t += len(curved_lift) - 1
            t += int(20 * sf)
            traj.append({"t": t, "action": np.concatenate([above_nut_pos, grip(angle, 1.)])})
            t += 1
            traj += [{"t": t + i, "action": np.concatenate([p, grip(angle, 1.)])} for i, p in enumerate(curved_transit)]
            t += len(curved_transit) - 1
            t += 1
            traj += [{"t": t + i, "action": np.concatenate([p, grip(angle, 1.)])} for i, p in enumerate(curved_insert)]
            t += len(curved_insert) - 1
            t += 1
            traj.append({"t": t, "action": np.concatenate([insert_pos, grip(angle, -1.)])})
            traj.append({"t": 600, "action": np.concatenate([insert_pos, grip(angle, -1.)])})
            self.trajectory = traj

    return RoundMultimodalWrapper, RoundSideScriptedPolicy


def create_round_collector_env(collector):
    """collector.create_robomimic_env() for NutAssemblyRound, with the round-nut wrapper."""
    RoundMultimodalWrapper, _ = collector_round_classes(collector)
    env_meta = collector.FileUtils.get_env_metadata_from_dataset("data/robomimic/datasets/square/mh/low_dim.hdf5")
    env_meta["env_kwargs"]["controller_configs"]['control_delta'] = False
    env_meta = round_env_meta(env_meta)
    collector.ObsUtils.initialize_obs_modality_mapping_from_dict({'low_dim': collector.OBS_KEYS})
    robomimic_env = collector.EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False,
                                                                render_offscreen=False, use_image_obs=False)
    env = collector.MultiStepWrapper(
        collector.VideoRecordingWrapper(
            RoundMultimodalWrapper(collector.RobomimicLowdimWrapper(env=robomimic_env, obs_keys=collector.OBS_KEYS,
                                                                    init_state=None)),
            video_recoder=collector.VideoRecorder.create_h264(fps=10, codec='h264', input_pix_fmt='rgb24', crf=22,
                                                              thread_type='FRAME', thread_count=1),
            file_path=None, steps_per_render=2),
        n_obs_steps=1, n_action_steps=1, max_episode_steps=600)
    return env, env_meta
